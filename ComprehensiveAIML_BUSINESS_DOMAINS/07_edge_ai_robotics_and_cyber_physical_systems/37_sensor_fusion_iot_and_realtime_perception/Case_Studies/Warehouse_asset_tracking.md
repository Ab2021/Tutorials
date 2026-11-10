---
case_id: "CASE-37-001"
case_title: "Door-Attributed Pallet Exit Verification in a Distribution Warehouse"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "draft"
content_status: "draft"
case_type: "streaming-systems"
domain: "Edge AI and cyber-physical systems / indoor asset tracking"
primary_pattern: "freshness gate"
secondary_patterns: ["association uncertainty", "telemetry lineage"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
source_status: "OPEN"
owner_role: "Warehouse systems engineering lead (illustrative role)"
source_index: "INDEX.md"
review_trigger: "Any change to dock-door layout, anchor or gateway placement, tag population, WMS exit-record contract, or audit protocol."
---

# Door-Attributed Pallet Exit Verification in a Distribution Warehouse

## 1. The One-Line Framing

Given a continuous stream of indoor position fixes for 2,400 tagged pallets and handling assets, decide at the instant a pallet crosses one of six dock-door apertures whether to emit a *verified exit* record naming that pallet and that door, for the outbound supervisor who currently reconciles doors and loads by hand at the end of the shift, subject to a 500 ms capture-to-decision latency budget, a 0.30 m root-mean-square (RMS) position budget driven by staging-lane adjacency, and a mandatory explicit *unknown* outcome when the evidence is insufficient, compared with a baseline of end-of-shift manual scan reconciliation.

**Evidence declaration.** This is an **illustrative design and evaluation exercise**. No warehouse, operator, vendor, deployment, incident, audit, or measured outcome is represented. Every site dimension, tag count, message rate, equipment price, labour rate, error rate, and audit result in this file is a constructed teaching value unless explicitly marked otherwise. Arithmetic that follows from those constructed inputs is *derived* and is checked by hand in the text; it is not a measurement of any system. Claims about standards or vendor capability that would require a datasheet are marked **OPEN** and are not load-bearing for the decision.

**Task category and decision frequency.** The system performs *streamed state estimation with a gated discrete decision*: it maintains a position and velocity estimate per asset, and each time an asset's estimated trajectory crosses a geometric aperture it fires a discrete, auditable event. The decision is event-driven rather than periodic — on a busy day the outbound area produces roughly 1,300 such crossings across six doors, with a burst of 40–60 within a two-minute window at shift change.

**Primary success condition in plain language.** When the system says "pallet P left through door D at time T", that statement is true, it is produced within half a second of the physical crossing, and it is accompanied by enough lineage to reconstruct why the system believed it. Everything else — dashboard, search UI, historical replay — is downstream of that one statement.

**Primary reusable pattern:** `freshness gate`. **Secondary patterns:** `association uncertainty`, `telemetry lineage`.

**The decisive trade-off, stated up front.** The obvious framing of this project is "locate my pallets". That framing pulls the design toward *average* accuracy over the whole floor, a large tag population, and a search experience — and the arithmetic in Section 6 shows it does not pay back: the labour it saves, at a plausible search rate, is smaller than the annualised cost of the infrastructure. The framing that does pay back is much narrower and, counter-intuitively, *easier* on space and *harder* on time. Six dock doors are 3.5 m wide and 6 m apart; assigning a pallet to a door needs only about 1 m of position accuracy, which the same hardware clears with two orders of magnitude of margin. But the decision must be made within 500 ms of the physical crossing and must be correct about *which of two adjacent pallets* moved, and that is where the engineering actually goes. The scarce resource is not accuracy in metres; it is the integrity of the timestamp and of the tag-to-asset binding.

**Why the simpler framing misses the point.** A pure "last known position" service, or a naive single-technology trilateration dashboard, answers "where was this pallet, roughly, at some point". It cannot answer "did the right pallet go out of the right door", because that question is about an *event boundary* with a deadline, an *identity* that may be wrong, and a *stale observation* that may be silently reused. Those three concerns are orthogonal to average position error, and a system optimised for average error can fail all three while looking excellent on a scatter plot.

```text
Frame A (rejected as the primary frame): "Where is pallet P right now?"
  -> optimises mean position error over the floor
  -> decision has no deadline, no identity risk, no audit trail
  -> value = search labour saved; small at realistic search rates

Frame B (adopted): "Did pallet P cross door D, provably, within 500 ms?"
  -> optimises capture-time integrity, tag-asset binding, and gate logic
  -> decision is discrete, auditable, and has a hard latency contract
  -> value = mis-shipment avoided; concentrated and measurable
```

The rest of this case builds Frame B and shows, with arithmetic, why it survives where Frame A does not.

## 2. Context & Background

### 2.1 Domain and physical setting

The setting is a single-site regional distribution warehouse of roughly 120 m by 80 m, or 9,600 m² of covered floor, serving retail replenishment. Goods arrive on inbound trailers, are received and put away into racking, are picked to outbound staging lanes, and leave through six dock doors onto outbound trailers. The asset population of interest is 2,400 items:

| Asset class | Count | Typical dwell | Motion profile | Tagging economics |
|---|---:|---|---|---|
| Outbound pallets in staging | 1,100 | 2–14 h | Mostly static, short 20–60 m moves by forklift | Cheap, high volume |
| Inbound pallets awaiting put-away | 600 | 1–6 h | Static then one long move | Cheap, high volume |
| Roll cages and totes | 450 | 4–30 h | Frequent short moves, tight clustering | Mid |
| Forklifts and powered pallet trucks | 180 | Continuous | 1.0–2.5 m/s, sharp turns, steel surroundings | Higher |
| High-value or regulated cages | 70 | Days | Static, high consequence if lost | Premium |

The spatial requirement is not uniform across these classes, and that non-uniformity is itself a design input. A lost roll cage is an inconvenience; a regulated cage leaving through the wrong door is an incident with a regulatory dimension. The engineering answer is a *tiered* accuracy and freshness contract rather than one number applied to everything.

### 2.2 Workflow before and after the decision

Before the decision: a picker completes a pick list against a store order, a pallet is built in a staging lane, and the pallet is labelled with a licence plate (a scannable identifier). A forklift driver collects the pallet and drives it to the door indicated on the load sheet. The driver scans the pallet and the door at the point of loading — in principle. After the decision: the warehouse management system (WMS) marks the pallet as shipped on a specific trailer, the trailer departs, and the store receives the pallet and reconciles against the advance shipping notice.

The failure the business actually experiences sits in the gap between "in principle" and "in practice": drivers skip scans under time pressure at shift change, scan the door but not the pallet, scan the pallet then load it on a different trailer, or scan correctly and the WMS rejects the message because the trailer was already closed. The result is a shipment record that disagrees with physical reality, discovered 6–30 hours later at the receiving store.

### 2.3 Stakeholders and who owns what

| Stakeholder | Relationship to the decision | What they will accept | What they will not accept |
|---|---|---|---|
| Outbound supervisor | Consumer of verified-exit records; resolves exceptions | A record that says "unknown, please scan" | A confident record that is wrong |
| Warehouse general manager | Budget owner for the infrastructure | Payback inside 24 months | A science project with no decision attached |
| WMS integration owner | Owns the exit-record contract and idempotency | Versioned schema, replayable events | Unbounded event volume, unordered writes |
| Health and safety lead | Owns anything mounted on vehicles or persons | Passive tags, no driver distraction | Screens in the cab, alarms that train drivers to ignore them |
| IT/OT security | Owns the network boundary and device identity | Mutual TLS, deny-by-default egress | Flat VLAN with anonymous devices |
| Site electrician / facilities | Physically installs anchors and power | Standard mounting, documented cable routes | Ad-hoc drilling above racking |
| Store operations | Downstream victim of mis-shipment | Fewer discrepancies | A new exception queue they must staff |

The decision owner for the *design* is the warehouse systems engineering lead. The decision owner for each individual *exit record* is the outbound supervisor, who may accept, reject, or convert a record to "unknown and re-scan". This separation matters: the system proposes, a human or an explicit policy commits, and the commit is journalled.

### 2.4 Baseline process and its documented weaknesses

The baseline is: manual scan at load, plus an end-of-shift reconciliation where the supervisor compares the WMS's expected door-to-trailer map against the scanned records and physically walks the staging area to find pallets that are recorded as loaded but still present, or absent but not recorded.

Its weaknesses, as a design matter rather than as a measured audit:

- **Coverage is voluntary.** Compliance depends on driver behaviour under time pressure, so the missingness is not random — it concentrates exactly at shift change and at the busiest doors, which is the worst possible pattern for a reconciliation that assumes missing-at-random.
- **Detection is late.** A discrepancy is found at best at end of shift, often at the receiving store, by which time the trailer has left and the corrective action is a re-delivery.
- **The scan is a claim, not evidence.** A driver who scans pallet P at door D has asserted a fact; nothing in the baseline verifies the physical co-location of P and D at that instant.
- **Reconciliation is labour.** The end-of-shift walk is a fixed cost per shift regardless of how many discrepancies exist.

A tracking system that simply digitises the same claim is not an improvement. The improvement, if there is one, must come from *independent physical evidence* that the pallet and the door coincided.

### 2.5 Scale, timing and physical boundaries

| Context field | Value used in this case | Status |
|---|---|---|
| Floor area | 9,600 m², 120 m × 80 m | Illustrative |
| Dock doors | 6, aperture 3.5 m, centre spacing 6.0 m | Illustrative |
| Racking height | 8.5 m, steel racking, dense metal | Illustrative |
| Tagged assets | 2,400 total (see class table) | Illustrative |
| Staging lane pitch | 2.4 m between pallet centres | Illustrative |
| Peak crossing rate | 60 crossings in 120 s at shift change | Illustrative |
| Daily crossings | ~1,300 across six doors | Illustrative |
| Shift pattern | Two shifts, 260 operating days per year | Illustrative |
| Operating temperature | 4–30 °C in ambient areas, −18 °C in the freezer annex | Illustrative |
| Network | Site Wi-Fi 6 plus a dedicated 5 GHz backhaul for gateways | Illustrative |
| Jurisdiction / regulatory | Not asserted; any retention or worker-monitoring obligation is OPEN | OPEN |

The freezer annex is called out deliberately. Battery behaviour, radio propagation, and condensation on radomes all change there, and a design validated only in ambient conditions has an unvalidated segment covering perhaps 15% of the tagged population. A case that ignores this would be teaching the wrong reflex.

### 2.6 Prior attempts and organisational dependencies

Three prior attempts are posited as *typical* rather than as any named site's history, and each fails for a structural reason worth naming:

1. **Passive RFID portals at the doors.** Portals read at the aperture reliably when the pallet passes, but they produce a read event with no identity confidence when two pallets pass within the read field together, and they cannot distinguish "approaching the door" from "through the door". The read is a *presence* signal at a *point*, not a *traversal* with direction.
2. **BLE beacon proximity.** Cheap and battery-friendly, but received signal strength is a terrible distance estimator in a metal-rich environment; the numbers in Section 6 show why a single RSSI-derived range at 20 m carries tens of percent relative error, and multipath in racking makes it worse.
3. **Manual cycle counting with a spreadsheet.** Accurate when executed, but a point-in-time sample that decays immediately, and it consumes the same labour the project is meant to release.

The dependency that constrains everything: the WMS exit-record contract is owned by another team with its own release cadence, and it currently has no concept of a *provisional* or *evidence-backed* exit. Adding one is an interface change, not a configuration change. That is a schedule risk, not a technical one, and Section 8 treats it as such.

## 3. The Business Question

### 3.1 The decision contract

| Decision contract | Content |
|---|---|
| Question | "Can the site replace voluntary driver scans with independent physical evidence for outbound pallet exit, and if so at which doors, for which asset classes, under what degraded-mode contract?" |
| Decision owner | Warehouse systems engineering lead, with the outbound supervisor as operational acceptor and the general manager as budget approver |
| Trigger and deadline | A mis-shipment review has raised the cost of outbound discrepancies; a decision is needed before the next capital cycle, six weeks out |
| Available actions | (a) no change; (b) expand manual auditing; (c) deploy full-floor location for all 2,400 assets; (d) deploy door-aperture verification only, for outbound pallets at all six doors; (e) deploy (d) at two doors as a pilot |
| Value | Avoided re-delivery, avoided credit notes, avoided receiving-store labour, and a reduced exception queue |
| Downside | Capital cost, installation disruption, a new failure surface that could *reduce* trust if it produces confident wrong records, and a risk that staff stop scanning because the system "knows" |
| Objective | Maximise the number of pallets whose door of exit is known, correctly, within the latency budget, subject to a precision floor |
| Hard constraints | Confident wrong records are worse than abstentions; capture-to-decision ≤ 500 ms; no new screens in vehicle cabs; WMS contract must remain backward compatible |
| Scope | Six dock doors, outbound pallet movement, one site, 260 operating days per year |
| Out of scope | Inbound receiving accuracy, pick-path optimisation, worker location, trailer interior tracking, cold-chain temperature evidence |

### 3.2 Objective, proxy, and where they diverge

The business objective is *reduced cost of outbound discrepancy*. The measurable proxy is the count of exit records that are both correct and produced inside the latency budget. These can diverge in two directions that the design must guard:

- **Coverage bought with precision.** A system can raise its record count by lowering its evidence threshold. Every additional record then carries more risk. Without a precision floor the metric is trivially gameable, which is why Section 9 makes precision a *guardrail* rather than a term in the objective.
- **Precision bought with abstention.** A system can raise precision to 100% by abstaining on everything hard. Without a coverage floor, the metric is gamed in the other direction. The two constraints must be evaluated jointly on the same audit sample.

The stated resolution: maximise verified coverage subject to precision on verified records being at least 0.995 on the audit sample, with the abstention path producing an explicit, staffed "re-scan required" task rather than silence.

### 3.3 What an expensive false decision looks like

Three distinct costs, and the design treats them differently:

| Error | Consequence | Cost class | Design response |
|---|---|---|---|
| False positive: pallet recorded as exited through door D, actually still staged | Trailer departs short; store discovers a shortage; re-delivery on a later trailer | High, recurring | Abstain rather than assert; require two independent crossing conditions |
| False negative: pallet exited but no record | Supervisor performs the manual scan they would have done anyway | Low, absorbed | Accept a coverage floor below 1.0 |
| Misattribution: pallet exited, but through the wrong door | Wrong trailer assumed; two trailers now have wrong manifests | Highest | Explicit unknown; never guess between adjacent doors |

The asymmetry is what makes this a *gated* decision rather than a *classification* problem. A classifier optimises total error; this system must weight a wrong confident record far above a missing one.

### 3.4 Why the question cannot be answered by a model score

No accuracy number, however good, answers the business question. The question asks whether a *contract* — capture-time semantics, tag-to-asset binding, gate logic, abstention, journal, and WMS interface — produces a defensible record. A system with 0.9 m mean accuracy but sloppy timestamps can be wrong at the door; a system with 0.15 m accuracy and an unverified tag binding can be confidently wrong about *which* pallet. Section 9 therefore evaluates the record, not the position estimator.

## 4. Assumptions & Constraints

### 4.1 Assumption and constraint register

| ID | Assumption / constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Every tag's binding to its asset is established by a witnessed scan at attach time and re-verified at each door | Data | Attach-and-verify procedure plus a reconciliation count | Unverified | Wrong-asset records; whole design invalid | Re-bind on next scan; quarantine records | Any tag detach event / tagging supervisor |
| A-02 | Sensors record a capture timestamp from their own counter, not the gateway's receive time | Data | Schema inspection and a delay-injection test | Unverified | Latency bias that grows with congestion | Edge-side re-stamping is prohibited; fix at source | Firmware change / integration lead |
| A-03 | The physical crossing of the aperture is detectable as a geometric crossing of a defined plane, robust to 0.2 m of vertical ambiguity | Physical | Video-referenced audit at 200 crossings | Unverified | Boundary chatter, duplicate events | Widen the hysteresis band; require two consecutive samples beyond the plane | Door layout change / perception lead |
| A-04 | Anchor positions are surveyed to ±0.05 m and re-surveyed after any racking reconfiguration | Infrastructure | Survey sheet with dated revision per anchor | Unverified | Geometry-dependent bias, worst near the mezzanine edge | Reduce claimed coverage, abstain in affected bays | Racking change / facilities |
| A-05 | Adjacent staging lanes hold pallets at least 2.4 m apart | Business | Layout drawing plus a spot check | Unverified | Association swaps between adjacent pallets | Lower the separation requirement only with a tighter tag | Layout revision / operations |
| A-06 | Tag battery life exceeds 18 months at the declared update rates | Infrastructure | Bench discharge at the freezer temperature | Unverified | Coverage hole appears silently as tags drop out | Raise the low-battery alert threshold; shorten the replacement cycle | Battery chemistry change / hardware |
| C-01 | Capture-to-decision must not exceed 500 ms at the 99th percentile | Latency | Load test at 3× the peak crossing rate | Scenario constraint | Supervisors stop trusting the live view | Degrade to batch reconciliation with a longer window | Any change to the decision's operational use |
| C-02 | Verified-record precision must be at least 0.995 on the audit sample | Quality | Audit against video and manual count | Scenario constraint | Confident wrong records destroy adoption | Raise the evidence threshold; accept lower coverage | Any change to the cost of a mis-shipment |
| C-03 | Verified coverage must be at least 0.90 of all outbound crossings | Quality | Same audit sample | Scenario constraint | The manual process survives in parallel and the saving does not materialise | Re-scope to the doors that clear it | Door mix change |
| C-04 | No new device may require driver interaction | Safety | Design review with the safety lead | Confirmed as a scenario rule | Redesign of the in-cab experience | None; this is a hard rule | Any proposed driver-facing feature |
| C-05 | Record retention on the edge is 48 h raw and 400 days for track snapshots | Data / cost | Storage sizing calculation | Scenario constraint | Replay-based diagnosis becomes impossible | Extend raw retention for the doors only | Audit finding / data owner |

### 4.2 The three assumptions most likely to reverse the design

**A-01, tag-to-asset binding.** This is the assumption that the case is really about. A position fix is evidence about a *tag*; the business decision is about a *pallet*. The bridge between them is a physical attachment that can be broken by a driver moving a tag, a pallet being re-labelled, or a tote being reused without a re-bind. If the binding is unreliable, no improvement in position accuracy helps, and the failure mode is the worst kind: a confident record about the wrong object. Every quantitative result in this file is therefore conditional on A-01, and Section 12 makes it a first-class failure mode rather than an appendix note.

**A-02, capture-time semantics.** If the only available timestamp is the gateway's receive time, the system's notion of "when" is really "when the network delivered it". Under congestion that quantity is dominated by queueing, not by physics. The worked example in Section 6 shows that a 1.2 s queue delay at a 1.2 m/s forklift speed displaces the attributed position by 1.44 m, which is 41% of the door aperture. That is not a rounding error; it is a boundary error.

**A-03, crossing detectability.** Turning a continuous position estimate into a discrete event requires a geometric predicate with hysteresis, because a noisy track that sits exactly on the plane will otherwise emit a burst of events. If the physical geometry is more complicated than a vertical plane — a sloped dock leveller, a pallet staged half-in the aperture — the predicate needs a documented exclusion rather than a cleverer rule.

### 4.3 Constraints that are not negotiable

C-01 through C-03 are the evaluation contract and are fixed before any measurement. C-04 is a safety rule with no engineering remedy: if a design requires driver interaction it is rejected, not tuned. C-05 bounds both cost and the ability to diagnose, and it is worth stating explicitly that 48 hours of raw retention is a *design limitation*, not a preference — it means a discrepancy discovered at day 3 on the receiving side cannot be debugged from raw data, only from track snapshots and the event journal.

### 4.4 What this case explicitly does not assume

- It does not assume any vendor's published accuracy figure transfers to a metal-rich racking environment. Vendor figures are typically open-air or line-of-sight; the case treats them as an upper bound on achievable quality and requires a site measurement.
- It does not assume the WMS can be changed on the same schedule.
- It does not assume staff will keep scanning once the system works. Section 12 treats the erosion of scanning as a *predicted* consequence with a mitigation, not as an unfortunate surprise.
- It does not assume the freezer annex behaves like the ambient floor.

## 5. Data, Tools & Method

### 5a. Data Specification

#### 5a.1 Unit of analysis and grains

Three grains coexist and must not be confused, because most of the interesting bugs live at their boundaries:

| Grain | Key | Cardinality per day | What it is | What it is not |
|---|---|---|---|---|
| Raw observation | (`tag_id`, `capture_seq`) | ~362.9 million | One ranging or beacon report from one tag to one infrastructure node | Not a position |
| Position fix | (`tag_id`, `fix_time`) | ~362.9 million at tag rate; ~207.4 million at 1 Hz after decimation | Fused estimate with covariance | Not an identity |
| Traversal event | (`asset_id`, `door_id`, `crossing_time`) | ~1,300 | A discrete claim that an asset crossed an aperture | Not yet a verified exit |
| Verified exit record | (`asset_id`, `door_id`, `exit_time`, `evidence_version`) | ≤1,300 | The commit-level output to the WMS | Not a shipment |

The **decision grain** is the traversal event. Everything upstream exists to make that event supportable; everything downstream consumes it. A common design error is to build the dashboard against position fixes, which makes the dashboard look correct while the event contract remains unevaluated.

#### 5a.2 Volume and window

| Quantity | Value | Derivation |
|---|---:|---|
| UWB-tagged assets at 5 Hz | 450 × 5 = 2,250 msg/s | Moving assets need finer sampling than static ones |
| BLE-tagged assets at 1 Hz | 1,950 × 1 = 1,950 msg/s | Static pallets need only slow refresh |
| Total observation rate | 4,200 msg/s | Sum |
| Observations per day | 4,200 × 86,400 = 362,880,000 | 24 h continuous |
| Payload per observation | 120 bytes | Fixed schema, see 5a.4 |
| Raw bytes per day | 43,545,600,000 B = 43.55 GB | 362,880,000 × 120 |
| Track snapshots per day | 2,400 × 86,400 = 207,360,000 | One per asset per second |
| Snapshot payload | 80 bytes | State vector plus covariance plus lineage |
| Snapshot bytes per day | 16,588,800,000 B = 16.59 GB | 207,360,000 × 80 |
| Evaluation window | 8 weeks (56 days) of which 2 weeks is the audit window | Long enough to span at least one layout change |

Note the ratio: raw telemetry is 2.6 times the volume of the derived snapshots while carrying almost none of the decision value. This asymmetry drives the retention decision in 5e.

#### 5a.3 Inputs and their availability

| Field group | Example fields | Units | Available at decision time? | Notes |
|---|---|---|---|---|
| Tag identity | `tag_id`, `tag_type`, `firmware` | — | Yes | Firmware version matters for filter behaviour |
| Capture timing | `capture_tick`, `tick_epoch`, `tick_rate_hz` | ticks | Yes | Tick plus epoch and rate, never a bare timestamp |
| Ranging | `range_m`, `rssi_dbm`, `quality`, `anchor_id` | m, dBm, unitless | Yes | `quality` is a device-reported confidence, not a calibrated probability |
| Anchor geometry | `anchor_xyz`, `survey_revision`, `mounting` | m, revision id | Yes, from a versioned table | Must be joined on *time*, not on latest |
| Asset binding | `asset_id`, `binding_id`, `bound_at`, `witness` | — | Yes, from the binding table | The critical join; see 5d |
| Door geometry | `door_id`, `plane_origin`, `plane_normal`, `aperture_width`, `geometry_revision` | m | Yes | Versioned alongside anchor survey |
| WMS expectation | `asset_id`, `expected_door`, `expected_trailer` | — | Yes, but **not** as a model input | Using it as an input would leak the answer |
| Ground truth (audit only) | `door_id`, `crossing_time`, `video_ref` | — | No, produced by audit | Must be joined strictly after the prediction |

#### 5a.4 Observation schema

The schema is deliberately narrow and versioned. Every field that a downstream consumer depends on is named; nothing is inferred from string parsing.

```text
Observation {
  schema_version : "obs/3"
  tag_id         : string          # stable device identity, rotated on re-provision
  capture_tick   : uint64          # device counter at the moment of transmission
  tick_epoch     : string          # boot epoch; invalidates ticks across reboot
  tick_rate_hz   : float           # nominal 5.0 or 1.0, with a declared tolerance
  anchor_id      : string | null   # null for beacon-only observations
  range_m        : float | null    # two-way time-of-flight derived
  rssi_dbm       : float | null    # reported for both technologies
  quality        : uint8           # 0-255 device confidence, uncalibrated
  gateway_id     : string
  receive_ns     : int64           # gateway clock; telemetry only, never physics
  payload_crc    : uint32
}
```

The rule that the schema encodes: `receive_ns` exists, is populated, and is used only for transport diagnostics. It is not a substitute for `capture_tick`. Making this explicit in the schema is cheaper than making it explicit in a design document that a future contributor will not read.

#### 5a.5 Outcome definition, maturity and adjudication

The label for the primary metric is the **audited crossing**: for a sampled set of pallets, a reviewer watches the door camera and records, to the nearest 0.5 s, the instant the pallet's leading edge crossed the aperture plane, plus which door it was. Two reviewers independently label 15% of the sample; disagreements are adjudicated by a third and the disagreement rate is reported as a label-quality diagnostic.

Label maturity matters: a crossing recorded by the system on day *d* can only be adjudicated once the corresponding video segment is available, which in this design is 24 h later. Any evaluation that runs before the video is available will silently drop the hardest cases, because hard cases are the ones where the record is disputed and therefore queued for review. The label pipeline must therefore have an explicit *pending* state and a fixed adjudication lag.

#### 5a.6 Population and exclusions

Included: pallets and roll cages crossing a door aperture while the site is operating. Excluded, with reasons:

| Exclusion | Reason | Consequence of excluding | Mitigation |
|---|---|---|---|
| Crossings during the planned 4-hour weekly floor wash | Anchors are covered, forklifts idle | Coverage estimate biased upward | Report separately; do not pool |
| Assets whose binding was created more than 30 days ago with no re-verification | A-01 risk | Excludes a real population that matters | Report as a separate stratum |
| Door 6 during weeks 3–4 (maintenance scaffolding) | Geometry changed mid-window | Would mix two geometries | Exclude the door-week, not the door |
| Trailer-interior movement after the pallet crosses | Out of scope by design | None | Not attempted |

Selection effects are named rather than smoothed: assets with recent bindings are easier to attribute, so a naive pooled precision figure overstates field precision. Section 9 stratifies by binding age for exactly this reason.

#### 5a.7 Join semantics

Two joins dominate the correctness of the whole analysis.

*Observation to asset.* `observation.tag_id → binding.tag_id` where `binding.bound_at <= observation.capture_tick_time < binding.unbound_at`. The join is **temporal and non-unique**: a tag can be re-bound, so a naive equi-join on `tag_id` produces duplicate rows and, worse, silently assigns an old observation to a new pallet. The correct join is a half-open interval join, and the row count of the interval table must equal the count of physical binding events.

*Anchor to geometry.* `observation.anchor_id → anchor_survey(anchor_id, valid_from, valid_to)`. Survey revisions change when racking moves. Joining on the latest survey rather than on the survey valid at capture time applies a future geometry to a past observation, which is the most common form of *geometric* leakage in this domain and produces beautifully wrong replay results.

#### 5a.8 Splits

| Split | Boundary | Purpose | Leakage control |
|---|---|---|---|
| Development | Days 1–28 | Filter tuning, gate threshold selection | No audit-window days |
| Audit | Days 43–56 | Precision and coverage estimation | Fully held out; thresholds frozen before day 43 |
| Layout-change probe | Days 29–42 | Deliberate racking relocation to test survey staleness handling | Analysed separately, never pooled |
| Door holdout | Door 5 excluded from tuning | Tests generalisation across door geometry | Used once, at the end |

The layout-change probe is unusual and deliberate: it converts a failure mode that would otherwise be hypothetical into a *reproduced* condition in a controlled window. Its results are reported as such, not folded into the headline.

### 5b. Tools & Stack

All components below are *proposed*; none is asserted to have been deployed. Naming is at the level of technology class and protocol, with the specific role each plays, because the mechanism is what transfers.

| Layer | Component | Role | Why this choice | What it must not do |
|---|---|---|---|---|
| Physical | UWB anchors (IEEE 802.15.4z class), 24 units | Two-way ranging to tags, position reference | Centimetre-class ranging in multipath is the only technology class with headroom against the 0.30 m budget | Must not be treated as surveyed until a dated survey exists |
| Physical | BLE gateways, 18 units | Presence and coarse zone for high-volume tags | Cheap, battery-friendly, adequate for "which half of the floor" | Must not be used to produce a metric position |
| Physical | UWB tags, 450 | Fine position for moving and high-value assets | 5 Hz update at vehicle speeds | Must not be attached without a witnessed binding |
| Physical | BLE tags, 1,950 | Presence and slow refresh for static pallets | 18-month battery target | Must not assert identity near two adjacent pallets |
| Edge | 2× edge compute nodes (fanless, 32 GB RAM, 8 cores) | Fusion, gating, event emission, 48 h buffer | Site survivability if the WAN link drops | Must not depend on cloud for the decision |
| Messaging | MQTT over TLS at the edge, mutual TLS per device | Device to edge transport | Standard, low overhead, per-device identity | No anonymous publish |
| Stream | Stateful stream processor (Flink-class or a purpose-built service) | Per-tag filter state, event-time windows, door gates | Keyed state with checkpointing | No wall-clock windowing |
| Storage (hot) | Time-series store | 48 h raw, 400 days snapshots | Range queries by tag and time | No unbounded raw retention |
| Storage (cold) | Object store | Audit video references, model and config versions | Cheap, immutable | No PII beyond what the audit needs |
| Integration | WMS adapter with an outbox pattern | Idempotent verified-exit delivery | Exactly-once effect via dedupe key | No synchronous call in the decision path |
| Observability | Structured event journal, per-stage latency histograms | Diagnosis and replay | Lineage from record back to observations | No sampling of denials |
| Client | Supervisor exception queue (desk, not cab) | Human resolution of "unknown" | Keeps humans out of the fast loop | No driver-facing UI (C-04) |

The two load-bearing choices are the *stateful stream processor* (because the decision needs per-tag history and event-time semantics) and the *outbox pattern* (because the WMS is a downstream system that will occasionally be unavailable, and a lost exit record is a business event, not a retryable log line).

### 5c. Method Summary

The pipeline is: observation → clock-domain normalisation → range-to-position solve → per-tag tracking → binding resolution → aperture crossing detection → evidence gate → verified exit or explicit unknown → journal → WMS.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Manual scan plus end-of-shift reconciliation (baseline) | Current process description | No capital cost; humans handle ambiguity well | Voluntary compliance, late detection, no physical evidence | Baseline; retained in parallel | Never fully retired; needed for the unknown path |
| Single-technology BLE RSSI trilateration | Path-loss arithmetic in Section 6 | Very low tag cost | Relative range error of ~42% at 4 dB RSSI noise, before multipath | Rejected analytically | If a site measurement shows RSSI noise far below 4 dB and open sight lines |
| Passive RFID portals only | Read-zone behaviour | Detects presence at the aperture cheaply | No direction, no traversal, ambiguous when two pallets pass together | Rejected for the primary decision | Could serve as a corroborating second signal |
| UWB ranging plus per-tag Kalman filter plus aperture gate | Ranging noise arithmetic and gate statistics | Meets the 0.30 m budget with ~1.6× margin | Anchor density and survey discipline are mandatory | **Chosen** (proposed) | If site-measured ranging noise exceeds 0.24 m RMS |
| UWB plus BLE plus video analytics fused | Multi-modality literature and engineering judgement | Highest robustness near occlusion | Cost, privacy review, and a second failure surface | Considered, not tested | If precision on occluded doors falls below 0.995 |
| Full-floor continuous tracking for all 2,400 assets | Cost model in Section 6 | Simple operational story | Fails the payback test at realistic search rates | Rejected on value, not on accuracy | If search frequency is far higher than assumed |

**Chosen method, stated as a mechanism.** Each tag maintains a four-dimensional state (position, velocity) in a constant-velocity model with process noise tuned to the asset class. Each UWB observation contributes a range measurement linearised about the current estimate; the filter update uses the range covariance, not a nominal constant. Every second, for each asset, the filter's predicted position is tested against each door plane; a crossing requires the *signed distance* to change sign between two consecutive fixes **and** both fixes to be inside the aperture's lateral extent, with a hysteresis band that suppresses re-entry chatter.

The discrete decision is then a three-way gate on the crossing candidate: if the posterior covariance is inside budget and the binding is current and the capture ages are inside the freshness limit, emit a verified exit; if any condition fails but the evidence is coherent, emit an *unknown* with a reason code; if the evidence is incoherent (a binding conflict, a reboot epoch change mid-crossing, a geometry revision mismatch), emit nothing to the WMS and raise an operational task.

### 5d. Data provenance, quality, exclusions and leakage controls

#### 5d.1 Source register

| Source ID | Content | Owner | Access basis | Extraction | Versioning | Limitation |
|---|---|---|---|---|---|---|
| `SRC-01` | Device observation stream, 56-day window | OT platform team | Site network capture | Continuous | Schema `obs/3`; firmware recorded per device | Firmware drift across the window |
| `SRC-02` | Anchor survey sheets with revision dates | Facilities | Site document store | Weekly | Dated revision per anchor | Survey accuracy of ±0.05 m asserted, not independently checked |
| `SRC-03` | Tag-to-asset binding log | Tagging supervisor | Operations system | Per binding event | Immutable append log with witness field | Witness field can be filled incorrectly |
| `SRC-04` | Door plane definitions | Warehouse systems | Configuration repository | Per layout change | Versioned, reviewed | Aperture lateral extent is a modelling choice, not a measured surface |
| `SRC-05` | WMS expected-exit records | WMS integration owner | Read replica | Daily | Snapshotted | Used for *evaluation only*, never as a model input |
| `SRC-06` | Door camera video | Security | Restricted | 24 h retention | Segment-per-hour | Retention shorter than the audit lag would break adjudication |
| `SRC-07` | Manual audit sheets | Audit reviewer | Restricted | Per audit batch | Signed per batch | Reviewer-dependent boundary calls |

#### 5d.2 Quality issues and their treatment

| Quality issue | Observed shape (illustrative) | Treatment | Why not drop the rows |
|---|---|---|---|
| Missing gateway timestamps | ~0.4% of observations | Recover from the adjacent packet; flag | Dropping biases the latency distribution toward the good network |
| Duplicate delivery | ~0.02%, always after a gateway reconnect | Deduplicate on (`tag_id`, `capture_tick`, `anchor_id`) | Duplicates would inflate coverage if counted as separate evidence |
| Out-of-order arrival | 3–8% within a 2 s window | Event-time processing with a 5 s watermark | Wall-clock ordering would attribute stale positions to later times |
| Capture epochs after reboot | ~1 per tag per month | Segment the filter state per epoch | A single epoch crossing would create a teleport |
| Binding gaps | ~1.2% of tag-time | Mark unknown, do not interpolate | Interpolation would fabricate identity |
| Anchor survey staleness | ~6% of anchor-days during the layout probe | Join on valid-time geometry | Latest-geometry join is leakage |

#### 5d.3 Leakage controls, stated as prohibitions

Four prohibitions, each of which corresponds to a way this analysis could produce a flattering but meaningless number:

1. **The WMS expectation must never enter the decision.** It is available at decision time and it is exactly the quantity being predicted. Using `expected_door` as a feature would give a beautiful precision curve and prove nothing. It is joined only in evaluation, after the record is frozen.
2. **Geometry must be versioned by valid time.** Applying the current anchor survey to a past observation lets a racking move "fix" historical errors.
3. **The audit label must not be derived from the system's own output.** If the reviewer is shown the system's claim while labelling video, agreement measures suggestion, not correctness. The labelling tool hides the claim.
4. **Thresholds must be frozen before the audit window.** Selecting the evidence threshold on the audit days and then reporting precision on the same days is selection on noise.

#### 5d.4 What is unavailable

The device firmware source for the UWB tags is assumed to be unavailable; behaviour at epoch boundaries is therefore characterised empirically rather than derived. The WMS's own reconciliation logic is available only as an output contract, not as code, so the analysis cannot quantify how the WMS would consume a provisional record — that is an interface negotiation, not a data question. No worker-motion data is collected, and no claim about productivity is supported.

### 5e. Reproducibility and method-selection ledger

| Artifact ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-01` | Site capture archive | `obs/3`, window 56 days | Primary analysis input | Restricted | Cannot be redistributed |
| `SRC-02` | Facilities document store | Revision-tagged | Geometry | Restricted | Survey accuracy unverified |
| `RUN-01` | Fusion service image + config digest | Build hash, config hash | Reproduce the pipeline behaviour | Internal | Requires hardware for real-time replay |
| `RUN-02` | Range-noise characterisation script | Script plus raw capture | Derive the 0.15 m ranging sigma | Internal | Site-specific; not portable |
| `RUN-03` | Gate-threshold sweep | Parameter grid plus logs | Justify the chi-square threshold | Internal | Frozen before audit |
| `CFG-01` | Filter and gate configuration | YAML, hash pinned | Exact parameters | Internal | — |
| `DOC-01` | This case | v1.0 | Decision record | Public in course | Illustrative inputs replace site values |

Reproducibility statement: the *logic* is reproducible from this document plus the configuration digest; the *numbers* are not reproducible because the inputs are constructed. A seed alone would not make the fusion reproducible either, since the pipeline is stateful and event-time dependent; the meaningful reproducibility unit is the (configuration digest, input window, code revision) triple.

**Method-selection ledger.** The choice of the UWB-plus-filter-plus-gate stack over BLE-only, RFID-portal-only, and full-fusion alternatives was made on derived ranging arithmetic and on the *shape* of the decision, not on a benchmark. The two facts that would flip the choice are: (i) a site ranging sigma materially above 0.24 m for the majority of the anchored area, which invalidates the position budget, and (ii) a decision that requires door attribution only at a coarse time resolution, in which case a portal-style presence signal becomes sufficient and much cheaper.
<!--APPEND-->
