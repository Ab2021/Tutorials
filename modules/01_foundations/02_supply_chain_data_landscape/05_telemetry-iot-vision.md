# Telemetry, IoT, and Computer Vision

> Module 02 — The Supply Chain Data Landscape
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02.01, 02.02, 02.03, 02.04
> Version 2026.5

---

## 1. Why this topic exists

Everything covered in Topics 01 to 04 is *declared* data. An order exists because a buyer
submitted it. A shipment record exists because a clerk or an interface created it. A master
record exists because someone maintained it. The data is a statement of intent or of
administration, and its grain, its keys, and its failure modes all follow from human process.

Telemetry is different in kind. A sensor reports a physical state. Nobody declared it. The
device measured a quantity, at an instant, at a location, and the record arrived without a
business meaning attached. Turning that record into something a planner can act on is a
modelling step, and that step is usually undocumented, which is why sensing projects
under-deliver relative to their hardware budgets.

Three examples show the shape of the problem.

**The excursion alert that fired 400 times.** A pharmaceutical distributor installed
temperature loggers on every outbound shipment and configured an alert whenever a reading left
2–8 °C. In the first quarter the system raised 412 alerts. Quality affairs investigated the
first 30 and found that 26 were single samples at a door opening, two were loggers packed
against a phase-change panel, one was a logger with an expired calibration, and one was a real
excursion. The team stopped reading the alerts. The system was technically correct at every
step and operationally useless, because no one had defined what an excursion *is* before
defining what to alert on.

**The 99% inventory accuracy claim.** A fashion retailer reported 99.2% inventory record
accuracy after an RFID rollout. An external audit, sampling 40 stock-keeping units (SKUs)
across six stores with a manual recount, measured 91.4%. The gap was not fraud. The operating
figure was computed from the RFID system's own reads against its own expected counts, in
stores where the read environment was favourable. The audit counted physical units on shelves
and in fitting rooms, including items the read zone did not cover.

**The dock camera that cost more than it saved.** A grocery distributor deployed computer
vision at 22 dock doors to record trailer arrivals and departures. Detection precision was
0.94, which sounded adequate. At 180 trailer movements per day across the site, 0.94 precision
produced roughly 11 false arrival events per day, each of which generated a yard check. The
labour cost of the false events exceeded the labour cost of the manual process the system
replaced, and nobody had computed that ratio before the purchase.

Each of these is the same error in different clothing: **a sensor reading was treated as a
business event without an explicit, tested mapping between the two.** The mapping is where the
analytical work lives, and it is the subject of this topic.

This topic sits after the systems-of-record, master-data, transaction-data, and external-signal
topics because sensing data only becomes usable once you can join it to identity. A temperature
series is worthless until you know which shipment, which lot, and which customer contract it
belongs to. That join is the hard part, and it depends entirely on the master data described in
Topic 02.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the evidence ladder from physical state to business decision, and **name** the
   specific failure that each rung introduces.
2. **Distinguish** a regulatory requirement from a contractual one in a cold-chain context, and
   **state** which document supplies the numerical limit in a given case.
3. **Compute** mean kinetic temperature (MKT) from a temperature profile by hand, **compare** it
   against a stated limit, and **explain** why it differs from the arithmetic mean.
4. **Specify** a sampling interval for a sensing use case from the shortest event you must
   detect, and **quantify** the detection probability that the interval implies.
5. **Evaluate** a vendor accuracy claim by identifying the test conditions it omits, for radio
   frequency identification (RFID), real-time location systems (RTLS), and computer vision.
6. **Size** the data volume and transmission cost of a sensing deployment, and **choose** between
   edge and cloud processing with a stated criterion.
7. **Design** a provenance and integrity scheme that would survive a dispute about what a sensor
   recorded.
8. **Price** a false-positive rate in labour and currency, and **use** that price to justify or
   reject a detection threshold.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a ladder of six rungs between the physical world and a decision. Every rung
adds interpretation, and every rung can break.

```text
  RUNG 6  DECISION            "quarantine the lot, notify QA, hold the batch"
              ^               requires: policy, threshold, authority, cost
              |
  RUNG 5  BUSINESS EVENT      "shipment SHP-4471 breached contract clause 7.2"
              ^               requires: identity join to shipment, lot, contract
              |
  RUNG 4  DERIVED FACT        "24 h MKT = 7.15 C; 6 h above 8.0 C; 15.8 degree-hours"
              ^               requires: episode definition, window, gap policy
              |
  RUNG 3  TIME SERIES         "96 hourly samples, 1 device, UTC, 0.1 C resolution"
              ^               requires: clock sync, unit normalisation, deduplication
              |
  RUNG 2  CALIBRATED SAMPLE   "(9.7 C, 2026-03-04T11:00Z, logger DEV-4471)"
              ^               requires: calibration in date, traceable reference, gain/offset
              |
  RUNG 1  TRANSDUCER SIGNAL   "1.842 V across a thermistor bridge"
              ^               requires: linearisation, excitation stability, ADC range
              |
  RUNG 0  PHYSICAL STATE      the actual product temperature at that point
                              never observed directly; only inferred
```

Read the ladder upward and downward. Upward is inference: each rung is a model, and each model
has an error. Downward is audit: given a decision, can you reconstruct the evidence that
justified it? A system that cannot be walked downward is not auditable, no matter how good the
upward inference is.

Five properties follow, and they explain most of what goes wrong.

**Property 1 — A sensor reports a state, not an event.** The state is continuous. The event is a
discrete claim about the state, produced by a rule. "Temperature exceeded 8 °C" is not in the
data. It is a rule applied to the data, with parameters (threshold, duration, hysteresis,
window) that someone chose. When the parameters are undocumented, the rule cannot be defended,
reproduced, or improved.

**Property 2 — The sampling interval bounds what you can ever know.** If the device samples every
15 minutes, an event lasting 9 minutes is detected with probability roughly 9/15 = 0.60 under
random phase, and when detected it appears as a single sample whose true duration is unknown.
No amount of downstream analytics recovers the information. The interval is a design decision
made once, usually by a firmware default, and it silently sets the ceiling on every later
question.

**Property 3 — Provenance is part of the datum.** A value of 9.7 °C is not evidence. It becomes
evidence when it carries a device identifier, a calibration status, a clock source, a location,
and a record of whether it was altered after capture. Strip any of those and you have a number.
In a dispute, a number loses.

**Property 4 — Coverage is geometric, not uniform.** Terrestrial AIS reaches roughly 40 nautical
miles from shore. Cellular reaches most populated land and little else. An RFID read zone is a
volume, not a doorway. A camera sees a cone, not a yard. Every sensing technology has a shape,
and the shape determines which parts of your process are observed and which are inferred from
absence.

**Property 5 — The cost of a false positive is borne by someone other than the buyer.** The team
that buys the camera is measured on deployment. The team that clears the false alarms is
measured on throughput. This split is why false-positive rates survive procurement, and why the
analyst who prices them is unpopular and correct.

### 3.1 The five questions for any sensing source

Before using any sensor feed, answer these in writing. If you cannot answer all five, you are
not ready to model on it.

| Question | Why it decides what you can build | Typical wrong answer |
|---|---|---|
| What physical quantity is measured, at what point? | Determines whether the reading represents the product, the package, or the air | "Temperature" — of what, where |
| How often, and with what resolution? | Sets the shortest detectable event | "Real time" — it never is |
| Who owns the device and the record? | Determines who can alter or withhold it | "We do" — often the carrier does |
| What is the calibration and clock status? | Determines whether a value is evidence | "It is digital, so it is accurate" |
| What does absence mean? | Distinguishes "no event" from "no coverage" | "No data means nothing happened" |

The fifth question is the one most often missed. A gap in an AIS track may mean the vessel was
out of terrestrial range, or that the transponder was switched off, or that the aggregator
dropped the poll. These are three different facts and they support three different conclusions.

> **Note:** Absence of a reading is not evidence of absence of the condition. A shipment with no
> temperature data is not a shipment that stayed cold. It is a shipment whose thermal history is
> unknown, and unknown is a distinct evidentiary state that a disposition decision must handle
> explicitly.

### 3.2 State, event, and the four kinds of derived fact

Most supply chain sensing reduces to four derived facts. Learn these and the vendor landscape
becomes easier to read, because every product is selling one of them.

| Derived fact | Definition | Typical source | Failure if the rule is naive |
|---|---|---|---|
| Excursion | The condition left its permitted band for longer than a stated duration | Temperature, humidity, shock logger | Single-sample triggers; no duration test |
| Dwell | Time an asset spent inside a geofence or at a facility | GNSS telematics, AIS, gate reads | First and last read bound the interval, not the truth |
| Occupancy | Whether a defined space contains a defined object | Camera, RFID portal, yard sensor | Double counts, occlusion, stale tracks |
| Condition | A quality state of the object, such as damaged or intact | Camera, vibration, shock | Class imbalance; the rare class is the one that matters |

Each of these has a threshold, a duration, and a scope. Write all three down. A dwell figure
without a geofence definition is not a number, it is an opinion with a decimal point.

## 4. Definitions and vocabulary

The vocabulary of sensing is borrowed from metrology, from radio engineering, and from
regulatory affairs, and the three borrowings conflict. "Accuracy" means one thing to a
metrologist and another to a warehouse manager. The table below fixes the meanings used in this
program.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Telemetry | Automated measurement transmitted from a remote asset to a receiving system | Used to mean any dashboard data | None |
| Sensor | A device containing one or more transducers plus signal conditioning and an interface | Treated as a source of truth | Device identifier |
| Sample | One measurement at one instant, with a value and a timestamp | Called a "reading" and treated as an event | Value plus unit plus UTC time |
| Sampling interval | Time between consecutive samples | Chosen by firmware default and never stated | Seconds or minutes |
| Accuracy | Closeness of a measurement to the true value | Confused with precision | Plus or minus degrees C |
| Precision | Repeatability of repeated measurements under identical conditions | Confused with accuracy | Plus or minus degrees C |
| Calibration | Comparison against a reference of known value, with a documented correction | Treated as a one-off purchase | Certificate with date and uncertainty |
| Metrological traceability | Unbroken chain of calibrations to a national or international standard | Assumed without evidence | Certificate chain |
| Mean kinetic temperature (MKT) | Single Arrhenius-weighted temperature producing the same degradation as a varying profile | Computed as an arithmetic mean | Degrees C, over a stated window |
| Activation energy | Arrhenius parameter expressing how strongly degradation depends on temperature | Left unstated in contracts | kJ/mol; default 83.144 |
| Excursion | A period in which a monitored condition leaves its permitted range | Declared on a single sample | Degrees C beyond limit, with duration |
| Stability budget | Cumulative tolerated exposure over shelf life, against which each excursion is charged | Treated as a per-event allowance | Degree-hours or degree-months |
| Cold chain | A temperature-controlled sequence from manufacture to administration | Used to mean "refrigerated truck" | Degree C band, for example 2-8 C |
| Data logger | Device that records samples over time for later retrieval | Assumed tamper-proof | Device identifier plus calibration record |
| Good Distribution Practice (GDP) | Regulatory guidance for wholesale distribution of medicinal products | Treated as a source of numerical temperature limits | Regulation citation |
| Telematics | Telemetry from a vehicle or mobile asset, including position | Used to mean fleet tracking only | Position plus vehicle bus signals |
| Automatic Identification System (AIS) | Maritime transponder broadcast of identity, position, course, and speed | Treated as a complete and continuous track | MMSI plus UTC timestamp |
| Terrestrial AIS | AIS received by shore stations, reaching roughly 40 nautical miles offshore | Assumed to cover the open ocean | Nautical miles |
| Satellite AIS | AIS received by satellites, with latency set by orbital pass geometry | Assumed real-time | Minutes to hours |
| Geofence | A defined geographic boundary used to infer an event from a position | Assumed exact and instantaneous | Polygon plus dwell threshold |
| Dwell time | Time an asset spends inside a geofence or at a facility | Computed from first and last read without stating it | Hours |
| Reefer | A refrigerated container with an integrated refrigeration unit | Used for any refrigerated transport | Container size and type code |
| RFID | Identification by a backscattered radio signal from a tag | Used loosely for any wireless tag | Electronic product code (EPC) or tag identifier |
| RAIN RFID | Passive ultra-high-frequency (UHF) RFID in the 860-960 MHz band | Treated as synonymous with all RFID | MHz band plus EPC |
| Read rate | Proportion of tags present that are successfully read in one pass | Quoted without the test conditions | Percent |
| Real-time location system (RTLS) | A system that estimates asset position continuously indoors | Confused with RFID | Metres of position error |
| Ultra-wideband (UWB) | Positioning radio using short pulses, giving sub-metre accuracy | Treated as centimetre-exact in every environment | Metres, plus latency in milliseconds |
| Confidence threshold | The score above which a detection is accepted | Tuned on the wrong cost ratio | Probability |
| False positive | A detection reported where no target exists | Dismissed as "noise" without a price | Count per period |
| Audit trail | Secure, time-stamped record of creation, modification, and deletion | Confused with an application log | Record plus retention period |
| ALCOA+ | Data-integrity principles: attributable, legible, contemporaneous, original, accurate, plus complete, consistent, enduring, available | Recited without being tested | Principle set |
| Digital twin | Computational model kept synchronised with a physical asset | Used for any simulation | Model plus synchronisation latency |
| EPCIS | GS1 standard for event data about objects moving through a chain | Used as a raw sensor-data format | Event schema |

Four of these deserve expansion, because they cause the most damage when confused.

**Accuracy versus precision.** A sensor can be precise and inaccurate: it returns 4.02, 4.02,
4.01, 4.03 C while the true value is 5.2 C. A calibration corrects this, and the corrected
series is good. A sensor can be accurate and imprecise: it averages correctly but scatters by
0.8 C, which no calibration fixes. For excursion detection, precision sets the false-alarm rate
and accuracy sets the bias. A logger with plus or minus 0.5 C accuracy near a 2 C lower limit
will declare excursions that are measurement artefacts, which is why the industry norm for
cold-chain loggers is tighter than the tolerance it polices.

**Excursion versus stability budget.** An excursion is an event. A stability budget is the
cumulative allowance that events consume. A product licensed for 24 months at 2-8 C may
tolerate a defined total exposure above 8 C across its life, derived from stability studies.
Treating each excursion against the full allowance, rather than against the remaining balance,
is a common and expensive error: it releases product that has already spent its budget, or
rejects product that has barely touched it.

**Read rate versus inventory accuracy.** Read rate is a property of a read event: of the tags in
the field, what fraction responded. Inventory accuracy is a property of a record set: of the
units that exist, what fraction the system knows about, in the right place. A 99% read rate in a
controlled doorway says nothing about the 4% of units sitting in a fitting room outside any read
zone. Vendors quote the first and imply the second.

**Digital twin versus simulation.** A simulation runs a scenario against assumed parameters. A
digital twin is kept synchronised with a physical asset by a data feed, and its value depends
entirely on the synchronisation latency and the fidelity of the state estimate. A twin fed by a
daily snapshot is a simulation with a delay. Section 9 reports what the 2025 literature finds
about how many deployed twins achieve genuine bidirectional synchronisation.

## 5. The mechanics, step by step

This section walks the ladder from the top of section 3 to the bottom, then back up. Each step
states what is recorded, what is lost, and the branch that decides everything downstream.

### Phase A — From physical state to a sample

**Step 1. The state exists and varies.**

The first question is not "what sensor" but "which temperature". A pallet of vials has a product
temperature, a vial-surface temperature, a shipper air temperature, and a container wall
temperature, and during a transhipment these can differ by 8 °C or more. The product temperature
is the one that matters clinically, and it is the hardest to measure without penetrating the
pack. Recorded: nothing yet. Lost: the spatial distribution, because a pallet is not isothermal.
Key branch: **is the sensor in the product, in the payload air, or in the container?** That
decides which contractual limit can legitimately be applied to it.

**Step 2. The transducer responds.**

A thermistor or resistance temperature detector converts temperature into a resistance, a bridge
converts that into a voltage, and an analogue-to-digital converter turns it into counts. Each
stage has an error and a drift. Self-heating from the excitation current is a real error at
0.01 °C resolution and irrelevant at 0.5 °C resolution. Recorded: raw counts or an engineering
value. Lost: the linearisation curve and the excitation stability, unless the device stores
them. Key branch: **is the raw value retained?** Retaining it lets you re-derive after a
calibration correction; discarding it makes every past reading hostage to the current
calibration.

**Step 3. The device samples and timestamps.**

The sampling interval is a firmware setting. The timestamp comes from a real-time clock that
drifts, typically by seconds to minutes per month, and is corrected only when the device
synchronises. A device that has been in a drawer for a year may be minutes off, which is enough
to move a reading across a transhipment boundary. Recorded: value, device-local time, sometimes
a sequence number. Lost: the relationship between device time and Coordinated Universal Time
(UTC) after the last synchronisation. Key branch: **does the record carry a UTC offset and a
sync status?** Without both, a disputed timestamp cannot be defended.

**Step 4. The device buffers.**

Devices store samples locally and transmit later. Buffer depth is finite. A device that wraps
its buffer after 8,000 samples loses the oldest data first, which is often the data covering the
loading event that caused the problem. Recorded: samples in flash or in a ring buffer. Lost:
anything beyond the buffer, silently. Key branch: **is buffer wrap-around flagged?** An unflagged
wrap looks identical to a complete record with an early start.

### Phase B — From sample to series

**Step 5. Transmission.**

The channel determines latency and cost. Cellular is cheap per message on land and absent at
sea. Satellite costs per message and per byte, so devices aggregate. LoRa aggregates many
sensors onto one gateway. Some devices are read by physical download at the destination, which
means the data exists only after the shipment is over and cannot support an in-transit
intervention. Recorded: transmitted samples with a gateway or network identifier. Lost: samples
transmitted during an outage, unless store-and-forward is implemented and verified. Key branch:
**is the data available in time to change a decision?** A logger downloaded at destination is a
post-mortem instrument, not a monitoring one.

**Step 6. Ingestion and normalisation.**

Units, scale, and time zone are reconciled here. A field documented as °C that arrives in °F
produces a plausible-looking series that is wrong by a factor of 1.8 plus an offset. A
resolution of 1 °C applied to a 2–8 °C band destroys the ability to detect a 0.5 °C trend.
Duplicate messages from a retrying gateway inflate counts and can double a dwell interval.
Recorded: normalised value, unit, timezone, source message identifier. Lost: the original
encoding, unless the raw payload is retained alongside. Key branch: **is the raw payload kept?**
Keep it; it is the only way to answer "what did the device actually send" six months later.

**Step 7. Gap handling.**

A gap has at least four possible meanings: the device was off; the device was out of coverage;
the gateway dropped the message; the platform rejected the message as invalid. Treating all four
as missing at random biases every downstream statistic, and treating a gap as "condition normal"
is a common and indefensible default. Recorded: an absence. Lost: the reason for it, unless the
device reports its own connectivity state. Key branch: **is the gap bounded by evidence on both
sides?** A gap bracketed by two normal readings differs from a gap at the end of a track.

**Step 8. Storage and retention.**

Retention is set by regulation, by contract, and by litigation risk, and the longest of the
three governs. Pharmaceutical distribution records are typically retained for years, not weeks.
Storing only aggregates destroys the ability to recompute an MKT under a different activation
energy, which is exactly what a dispute may require. Recorded: series plus metadata. Lost:
everything outside the retention window, by design. Key branch: **can you recompute the derived
fact from the retained data?** If not, the derived fact is the record, and it is unauditable.

### Phase C — From series to derived fact

**Step 9. Episode definition.**

An excursion rule needs four parameters: a threshold, a minimum duration, a hysteresis band to
prevent chatter at the boundary, and a minimum number of samples. Most systems implement only
the first. The result is that a single sample at 8.01 °C creates an excursion, and a genuine
two-hour event that crosses the threshold slowly may create none.

| Parameter | Typical value | What it controls | Failure if omitted |
|---|---|---|---|
| Threshold | 8.0 °C | Which readings count as out of band | None; always present |
| Minimum duration | 30 min | Whether a transient counts | Single-sample alerts |
| Hysteresis | 0.5 °C | Re-triggering near the boundary | Alert storms |
| Minimum samples | 2 | Robustness to a single bad sample | One corrupt sample creates an event |

**Step 10. Window selection.**

MKT is computed over a defined window, not over the whole shipment. Current pharmacopoeial
guidance specifies a 30-day window for controlled room temperature products and a 24-hour window
for controlled cold temperature products, counted back from and including the highest excursion
temperature. Computing MKT over a 96-hour journey and comparing it to a 24-hour limit is not a
conservative choice; it is a different statistic. Key branch: **which window does the contract
specify, and does it match the regulation?** Often they differ, and the contract governs the
commercial outcome.

**Step 11. Identity join.**

The series must attach to a shipment, a lot, and a contract. The device identifier is not the
shipment identifier. The join is usually a manual association at packing, recorded on a form or
scanned into a system, and it is the weakest link in the chain. A logger assigned to the wrong
pallet produces a confident, well-formed, wrong answer. Key branch: **is the association recorded
at pack time with a scan, or reconstructed later from a list?** Reconstruction is where disputes
are lost.

### Phase D — From fact to business event

**Step 12. Rule application and disposition.**

The derived fact meets a limit. The limit comes from one of three documents, and they are not
interchangeable: the marketing authorisation or product stability data, the quality agreement
between the parties, and the carrier's service commitment. A breach of the third is a service
issue, a breach of the first is a regulatory issue, and a breach of the second is a commercial
one.

**Step 13. Alert routing and acknowledgement.**

An alert is a work item with an owner, a service level, and a closure record. Alerting without
acknowledgement tracking produces the 412-alert quarter described in section 1. The design
question is not "how do we detect more" but "what will a human do differently, within what time,
and who is accountable when they do not".

### Phase E — From event to record

**Step 14. Audit trail and dispute.**

The record must show what was measured, by which device, calibrated when, transmitted how,
stored where, and whether anyone altered it. Under pharmaceutical good practice, records must be
attributable, legible, contemporaneous, original, and accurate, with the additional attributes
of completeness, consistency, endurance, and availability. A supplier portal that lets a carrier
edit a temperature value after the fact, with no trail, fails every one of those tests. Key
branch: **can the record be altered by a party with an interest in the outcome?** If yes, it is
a claim, not evidence.

### 5.1 Cold chain: placement, mapping, and validated duration

**Placement follows mapping, not convenience.** A storage area is temperature-mapped before use,
and monitoring devices are placed at the locations the mapping identified as extremes, not at
the door because the cable reaches. In a shipper, the analogue is a validation study that places
loggers at the payload centre, the payload edge, and the ambient reference, and demonstrates
that the payload centre stays in band for the qualified duration.

**Duration is validated, not assumed.** A passive shipper has a validated duration under stated
ambient conditions. A 72-hour qualification at 30 °C ambient says nothing about 96 hours at
38 °C. The excursion in section 6 happens precisely because the shipper was used beyond its
validated duration.

### 5.2 Telematics and AIS: what each actually reports

| Source | What it reports | Typical update interval | Coverage limit | Constraint on use |
|---|---|---|---|---|
| Container IoT device | Position, temperature, humidity, door events | 15 min on land; 1–24 h at sea | Cellular near shore; satellite offshore | Carrier owns the device and the record |
| Carrier vessel API | Vessel position, estimated arrival, port events | Minutes to hours, carrier-dependent | Only vessels the carrier operates or charts | Contractual terms; not all partners |
| Terrestrial AIS | Vessel identity, position, course, speed | About 2 min | Roughly 40 nautical miles from shore | Public broadcast, but aggregator licences restrict redistribution |
| Satellite AIS | Same fields, collected by satellite | 30 min to 4 h, latitude-dependent | Global, with per-pass latency | Latency is structural, not a fault |
| Truck telematics | Position, ignition, speed, harsh events, fuel | 30 s to 5 min | Cellular coverage | Driver data is personal data in many jurisdictions |
| Trailer tracking | Position, door open and close, cargo sensor | Minutes to hourly | Cellular; blind inside metal warehouses | Battery life bounds the interval |

Two consequences deserve emphasis. First, the update interval is a property of the mode, not a
shortcoming of the vendor: an ocean leg reported every 24 hours is normal, and an analytics
design that assumes minute-level ocean visibility will produce false "no movement" exceptions.
Second, the record is frequently owned by a counterparty. A carrier that supplies position data
through its own portal controls what you can query, how long it is retained, and whether you may
redistribute it.

> **Note:** A gap in an AIS track is ambiguous between "out of terrestrial range", "transponder
> off", and "aggregator poll missed". These have different legal and operational meanings. Any
> model that treats a gap as a single category of missingness is modelling the data pipeline,
> not the vessel.

### 5.3 RFID and RTLS: read physics before read rates

Passive UHF tags work by backscatter: the reader illuminates the tag, and the tag reflects a
modulated fraction of that energy. The returned signal is orders of magnitude weaker than the
transmitted one, which makes the read environment the dominant variable. Three physical effects
govern performance:

- **Metal** reflects and detunes. A tag in direct contact with metal can become unreadable.
  On-metal designs use a spacer or a patch antenna, and still degrade when adjacent metal comes
  within tens of centimetres.
- **Liquid** absorbs UHF energy. Water-rich products reduce read range, and a pallet of bottled
  goods behaves very differently from a pallet of cardboard.
- **Geometry** creates nulls. Multipath cancellation produces dead spots that move when racks
  move, so a read rate measured at installation is not the read rate after a layout change.

Active RTLS technologies sidestep some of this. Ultra-wideband positioning achieves sub-metre
accuracy in controlled tests, with reported mean absolute errors around 0.1 m for vehicle
tracking, degrading to 0.3–0.4 m when a human body blocks the direct path. It requires anchor
infrastructure, and its cost scales with the number of anchors and the site geometry rather than
with the number of tags.

### 5.4 Computer vision: the pipeline and where the cost sits

A vision deployment is five stages, and the labelling cost sits in the third.

```text
  camera frame --> detector --> tracker --> event rule --> business fact
  (30 fps)        (per frame)   (per track)  (per episode)  (per decision)
       |              |             |             |
       |              |             |             +-- dwell, occupancy, count
       |              |             +-- identity persistence across frames
       |              +-- labelled examples, thousands per class
       +-- exposure, glare, occlusion, night, weather
```

The detector is the visible cost. The labelling is the hidden one: a damage-detection model
needs examples of each damage type under each lighting condition, and the rare classes are
precisely the ones with few examples. Reported precision in the parcel-damage literature
clusters between 0.91 and 0.97 at high-recall operating points, which implies false-positive
rates of roughly 3–9% on the positive class. Sections 6 and 12 use that range to price a
threshold. Two vision use cases behave very differently and should not be specified together.

| Use case | What is detected | Error that hurts | Tolerance for error |
|---|---|---|---|
| Dock door occupancy | Trailer present at a door | False occupancy blocks a door | Low; false positives are cheap to clear |
| Trailer counting | Number of trailers in a yard | Miscount distorts yard capacity | Moderate; aggregated over a shift |
| Damage detection | A damaged case or parcel | Missed damage reaches a customer | Low for misses, high for false alarms |
| Yard safety | Person in a vehicle zone | Missed detection is an injury | Very low tolerance for misses |

## 6. Worked example

A cold-chain shipment with a temperature excursion, computed end to end. All figures are
illustrative and internally consistent.

### 6.1 The shipment and the question

Twelve pallets of a 2–8 °C biologic travel Dublin to Chicago, through a Frankfurt transhipment,
door to door in 96 hours. The shipper is passive: phase-change panels in a validated insulated
container, qualified for 72 hours at 30 °C ambient. The product's unit value is EUR 180, and
each pallet carries 480 units.

The question is narrow and commercial: **can this shipment be released, or must it be rejected
and replaced?** Everything below exists to answer that question, and every number is needed.

### 6.2 The evidence available

Four loggers travel with the shipment: three inside the payload of pallets 4, 9, and 11, and one
on the inside face of the container lid, which is an ambient reference and not a product
temperature. Each samples hourly at 0.1 °C resolution. Calibration certificates are in date,
issued 7 months earlier with an uncertainty of ±0.3 °C.

At Frankfurt the container sits on the tarmac for 4 hours 50 minutes at 24 °C ambient. The
phase-change panels are already 61 hours into their 72-hour validated duration. Logger A, at the
centre of pallet 4, records a rising profile; the other payload loggers rise less; the lid
logger rises most.

### 6.3 The 24-hour window

The quality agreement specifies MKT over any 24-hour period, with a limit of 8.0 °C, and no
single reading above 15.0 °C. The contract also charges cumulative exposure above 8.0 °C against
a 72 degree-hour stability budget.

The window is the 24 hours containing the peak, counted back from and including the peak hour.
Logger A's readings in that window are grouped below by temperature; several hours share a
value, so the table lists distinct temperatures with their counts rather than 24 rows.

| Temperature (°C) | Hours | Cumulative hours |
|---|---:|---:|
| 5.0 | 7 | 7 |
| 5.4 | 9 | 16 |
| 6.2 | 1 | 17 |
| 7.4 | 1 | 18 |
| 8.2 | 1 | 19 |
| 8.6 | 1 | 20 |
| 9.7 | 1 | 21 |
| 11.4 | 1 | 22 |
| 12.1 | 1 | 23 |
| 13.8 | 1 | 24 |

Six consecutive hourly readings sit above 8.0 °C, peaking at 13.8 °C. The excursion is six
hours long, which is 6.25% of the 96-hour journey.

### 6.4 The MKT arithmetic

MKT is an Arrhenius-weighted mean. Each sample is converted to an absolute temperature, and the
weight of a sample grows exponentially with its temperature, because degradation kinetics do.

MKT = (ΔH / R) / ( −ln( (1/n) × Σ exp( −ΔH / (R × T_i) ) ) )

With ΔH = 83.144 kJ/mol and R = 0.0083144 kJ/(mol·K), the ratio ΔH/R is exactly 10,000 K. A
sample at temperature T_i contributes `exp(−10,000 / T_i)`, with T_i in kelvin.

| °C | Hours n | T (K) | 10,000 / T | exp(−10,000/T) × 10^16 | n × term |
|---|---:|---:|---:|---:|---:|
| 5.0 | 7 | 278.15 | 35.9518 | 2.4341 | 17.0387 |
| 5.4 | 9 | 278.55 | 35.9002 | 2.5630 | 23.0670 |
| 6.2 | 1 | 279.35 | 35.7974 | 2.8405 | 2.8405 |
| 7.4 | 1 | 280.55 | 35.6443 | 3.3102 | 3.3102 |
| 8.2 | 1 | 281.35 | 35.5429 | 3.6634 | 3.6634 |
| 8.6 | 1 | 281.75 | 35.4925 | 3.8527 | 3.8527 |
| 9.7 | 1 | 282.85 | 35.3544 | 4.4235 | 4.4235 |
| 11.4 | 1 | 284.55 | 35.1432 | 5.4638 | 5.4638 |
| 12.1 | 1 | 285.25 | 35.0569 | 5.9565 | 5.9565 |
| 13.8 | 1 | 286.95 | 34.8493 | 7.3303 | 7.3303 |

Sum of the weighted terms: 76.9466 × 10^−16 over n = 24 hours.

Mean = 76.9466 / 24 = 3.2061 × 10^−16

ln(3.2061 × 10^−16) = −35.6763

MKT = 10,000 / 35.6763 = 280.30 K = **7.15 °C**

### 6.5 Why MKT differs from the arithmetic mean

The arithmetic mean of the same 24 readings is 161.0 / 24 = 6.71 °C. MKT is 0.44 °C higher.

The difference is not an error. It is the point of the statistic. Arrhenius weighting gives the
warm hours disproportionate influence: the six hours above 8 °C are 25% of the window but
contribute 40% of the summed exponential mass (30.69 of 76.95). The arithmetic mean treats an
hour at 13.8 °C as equivalent to an hour at 5.0 °C, which the degradation chemistry does not.

By Jensen's inequality, MKT is always greater than or equal to the arithmetic mean whenever the
profile varies. The two coincide only for a constant temperature. An analyst who reports the
arithmetic mean while the contract specifies MKT will understate exposure every time.

### 6.6 Duration dominates peak

The same six excursion hours, redistributed, produce very different MKT values. This is the
single most useful sensitivity in cold-chain work.

| Scenario | Excursion composition | 24 h MKT (°C) | Verdict against 8.0 °C |
|---|---|---:|---|
| Actual profile | Ramp 8.2, 8.6, 9.7, 11.4, 12.1, 13.8 | 7.15 | Pass |
| Short peak | 3 h at 13.8, 3 h at 5.4 | 7.04 | Pass |
| Sustained peak | 6 h at 13.8 | 8.40 | Fail |

A brief spike to 13.8 °C is survivable. Six hours at the same temperature is not. Any alerting
rule built on peak temperature alone would treat these two events identically, and would be
wrong in both directions: over-reacting to the spike and under-reacting to the sustained event.

### 6.7 The contract comparison

Three tests, three answers.

| Test | Limit | Observed | Result |
|---|---|---|---|
| 24 h MKT | ≤ 8.0 °C | 7.15 °C | Pass, margin 0.85 °C |
| Single reading | ≤ 15.0 °C | 13.8 °C | Pass, margin 1.2 °C |
| Cumulative exposure | ≤ 72 degree-hours | 15.8 degree-hours | Charges 22% of budget |

Degree-hours are computed as the sum of positive excess above 8.0 °C:
0.6 + 3.4 + 5.8 + 4.1 + 1.7 + 0.2 = 15.8 °C·h.

Note the disagreement between the metrics. MKT passes comfortably, and the stability budget
takes a 22% hit from a single event. A product with four such events in its life exhausts the
budget while passing every individual MKT test. This is why the budget, not the episode, is the
right unit of account.

The lid logger, meanwhile, peaked at 17.2 °C. It is not a product temperature and must not be
compared to the product limit. If the quality agreement said "any logger in the shipper", the
same shipment would fail on a reading that never touched the product. Read the scope clause
before you read the numbers.

### 6.8 The disposition decision and its cost

The shipment value is 12 pallets × 480 units × EUR 180 = EUR 1,036,800.

| Option | Direct cost | Time | What it resolves |
|---|---|---:|---|
| Release without assessment | EUR 0 | 0 d | Nothing; carries unquantified quality risk |
| QA assessment, then decide | EUR 4,200 | 10 d | Whether the excursion is within the stability data |
| Stability testing on retained samples | EUR 20,300 | 21 d | Whether the specific lot degraded |
| Reject and reship | EUR 1,061,700 | 14 d | Certainty, at full product cost |

The stability test figure includes EUR 9,500 of laboratory work and EUR 10,800 of product
consumed (60 units at EUR 180). The reject figure is product EUR 1,036,800 plus freight
EUR 18,400 plus expedite premium EUR 6,500.

The decision-theoretic reading is simple. The assessment costs 4,200 / 1,061,700 = 0.40% of the
value at stake. It is worth buying whenever it has more than a 0.4% chance of changing the
disposition. The stability test costs 20,300 / 1,061,700 = 1.91% and 21 days; it is worth buying
when the probability of a changed disposition exceeds 1.9% and the delay is tolerable.

Here the assessment is the obvious purchase, and the decision it produces is release with a
budget charge: the lot ships, the 22% budget consumption is recorded against the lot, and the
remaining 56.2 degree-hours are available for the rest of its life.

## 7. Code, query, or calculation

The calculation that matters for this topic is the MKT computation, and it is worth writing
once, plainly, so that you can check any tool against it. The block below is deliberately
small: it takes a list of hourly temperatures and returns the MKT in degrees Celsius, with the
window and activation energy explicit.

```python
import math

DELTA_H_J = 83_144.0        # J/mol, default per USP <1079.2>
R = 8.3144                  # J/(mol K)

def mkt_celsius(readings_c, activation_j=DELTA_H_J):
    """readings_c: list of hourly temperatures in degrees Celsius, equal weight."""
    if not readings_c:
        raise ValueError("empty window")
    n = len(readings_c)
    total = 0.0
    for t_c in readings_c:
        t_k = t_c + 273.15
        total += math.exp(-activation_j / (R * t_k))
    mean_term = total / n
    mkt_k = (activation_j / R) / (-math.log(mean_term))
    return mkt_k - 273.15

# Section 6 profile, one value per hour
profile = [5.0]*7 + [5.4]*9 + [6.2, 7.4, 8.2, 8.6, 9.7, 11.4, 12.1, 13.8]
print(round(mkt_celsius(profile), 2))   # 7.15
```

Three checks before you trust any MKT implementation, including your own.

- **Constant input returns the constant.** A profile of 24 identical readings at 5.0 °C must
  return 5.00 °C. If it returns 4.99 or 5.01, the implementation has a units error between
  joules and kilojoules, which shifts ΔH/R by a factor of 1,000.
- **Kelvin, not Celsius.** Passing Celsius into the exponent produces a number, and the number
  is meaningless. This is the single most common implementation defect.
- **MKT is never below the arithmetic mean.** If it is, the sum has been inverted or the sign on
  the logarithm dropped.

The activation energy is a contractual and scientific parameter, not a constant of nature. At
83.144 kJ/mol the section 6 profile gives 7.15 °C. Raising ΔH weights warm samples more heavily
and raises MKT; lowering it pulls MKT toward the arithmetic mean. When a supplier and a customer
disagree about an excursion, check that they used the same ΔH before arguing about the data.

## 8. Data, standards, and regulatory context

Sensing data is governed at three layers, and confusing them is the most common source of
compliance error. Metrology governs whether a value is trustworthy. Product regulation governs
what must be recorded and retained. Contract governs what happens when a limit is breached.

### 8.1 Metrology and qualification

| Standard or practice | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO/IEC 17025 | Calibration and testing laboratory competence | Whether a calibration certificate is traceable and defensible | B |
| GMP qualification and validation (EudraLex Vol. 4, Annex 15) | Equipment fitness for purpose | Whether a monitoring system is installed and proved before use | B |
| GAMP 5 | Computerised system validation by risk | How much validation evidence a monitoring system needs | C |
| Temperature mapping before use | Storage area characterisation | Where monitoring devices must be placed | B |

The practical consequence: a logger is not qualified because it is new, and a monitoring system
is not validated because it was installed by a vendor. The evidence is a document set, and its
absence is a finding in an inspection regardless of whether the data is correct.

### 8.2 Cold chain: regulation, pharmacopoeia, and contract

| Instrument | Jurisdiction | What it actually requires | Grade |
|---|---|---|---|
| EU GDP Guidelines 2013/C 343/01 | European Union | Temperature mapping, calibrated monitoring traceable to a national standard, annual recalibration of transport equipment, risk-based qualification | B |
| EU GMP Annex 11 (computerised systems) | European Union | Validation, data secured against modification, audit trail, backup retained | B |
| USP <1079> | United States, pharmacopoeial | Good storage and distribution practices for drug products | B |
| USP <1079.2> | United States, pharmacopoeial | MKT definition, activation energy default, 30-day and 24-hour windows, prohibited practices | B |
| USP <1079.3> | United States, pharmacopoeial | Monitoring devices and their qualification | B |
| IATA Temperature Control Regulations (TCR) | Air transport | Shipper and carrier obligations, time-and-temperature labels, acceptance checklists | B |
| IATA CEIV Pharma | Air transport, voluntary certification | Audited capability against the TCR checklist, including sensor calibration | C |
| WHO TRS 961 Annex 9 | International guidance | Model guidance on storage and transport of time- and temperature-sensitive products | B |
| 21 CFR Part 11 | United States | Electronic records and signatures; audit trails, access control, record retention | B |

Two distinctions are worth stating explicitly, because they are the source of most disputes.

**Regulation sets the method, the product licence sets the number.** EU GDP requires calibrated,
traceable monitoring and defined calibration intervals. It does not itself declare that 2–8 °C
is the limit, nor how long an excursion may last. Those come from the marketing authorisation
and the stability data behind it. An analyst who cites GDP as the source of a numerical limit
has misread the document.

**The contract sets the consequence.** A quality agreement may impose a stricter limit than the
licence, allocate the cost of investigation, and define the disposition authority. When the
contract and the regulation differ, compliance requires meeting both and the commercial outcome
follows the contract.

> **Note:** The 2025 revision of USP <1079.2> tightened the MKT method, fixing the 24-hour window
> for controlled cold temperature products and prohibiting the practice of offsetting a warm
> excursion with subsequent cold storage. An internal procedure written before that revision may
> compute a defensible-looking MKT that no longer matches the current method.

### 8.3 Data integrity for device-originated records

| Requirement | Source | What it means for a sensor record | Grade |
|---|---|---|---|
| Secure, computer-generated, time-stamped audit trail | 21 CFR 11.10(e) | Every create, modify, and delete is recorded, and prior values are not obscured | B |
| Accurate and complete copies | 21 CFR 11.10(b) | Export must reproduce the record, including metadata | B |
| Device checks | 21 CFR 11.10(h) | The system validates the source of data input, which for telemetry means device identity | B |
| ALCOA+ attributes | FDA data-integrity guidance and MHRA/PIC-S practice | Attributable, legible, contemporaneous, original, accurate, plus complete, consistent, enduring, available | B |
| Metadata including instrument identity | FDA CGMP data-integrity Q&A, 2018 | The device identifier and acquisition timestamp are part of the record, not decoration | B |
| Time and timezone accuracy | FDA electronic records guidance, 2024 | Clock accuracy and timezone documentation must be controlled | B |

The 2024 FDA guidance on electronic systems in clinical investigations is the clearest statement
of how device-originated data is treated: the device is the originator of the data, and the
metadata identifying it, with acquisition time and timezone, belongs in the durable record. That
principle transfers directly to supply chain sensing even where the specific guidance does not
apply.

### 8.4 Identification and event standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 EPCIS 2.0 | Event data about objects | Object, aggregation, transaction, transformation, and association events, now with optional sensor elements | B |
| GS1 CBV 2.0 | Event vocabulary | Sensor property types such as temperature and humidity, with units of measure | B |
| ISO 6346 | Container identification | Owner code, serial, and check digit on every container | B |
| DCSA IoT event standard | Carrier container telemetry | A common shape for position and sensor events from ocean carriers | C |
| SOLAS V/19 | Maritime safety | AIS carriage requirements for vessels on international voyages | B |

EPCIS 2.0 is the only widely adopted standard that lets a sensor reading travel as part of an
identified business event rather than as a raw time series. Its guidance is explicit that EPCIS
is not a raw sensor-data dump: it expects aggregated, business-oriented values with a pointer to
raw data where needed. That design choice is a useful discipline, because it forces the question
of what the reading means before the reading is stored.

### 8.5 Privacy and contractual limits

| Constraint | Where it bites | Practical effect | Grade |
|---|---|---|---|
| GDPR and equivalent personal-data law | Driver location, in-cab camera, tachograph | Lawful basis, retention limits, works council consultation | B |
| AIS licence terms | Redistribution of vessel positions | Internal use permitted; resale and derivative products usually restricted | B |
| Carrier data terms | Container and vessel events from carrier portals | Query limits, retention, no redistribution, no bulk export | C |
| Contractual confidentiality | Customer shipment data held by a logistics provider | Aggregation and benchmarking may breach the agreement | C |

The AIS licence point is frequently missed. AIS is a public broadcast, but the aggregated feeds
that make it useful are licensed products, and the licence typically forbids redistribution and
derivative commercial use. A model trained on a licensed feed may itself be a derivative work.

### 8.6 What is convention rather than requirement

- A 15-minute sampling interval is a habit inherited from logger firmware, not a regulatory
  requirement. The requirement is that the interval be justified against the shortest event you
  must detect.
- A ±0.5 °C logger accuracy is an industry norm, not a legal limit. The defensible figure is
  derived from the tolerance you are policing.
- An annual calibration interval is a common reading of GDP for transport equipment, not a
  universal rule for every device in every jurisdiction.

## 9. Industry practice and case evidence

### 9.1 Tag economics and adoption

The RAIN Alliance, the industry body for passive UHF RFID, reported 42.7 billion tag chip
shipments in 2025, down from a record 52.8 billion in 2024. The decline is attributed to a
semiconductor inventory correction, tariff uncertainty affecting United States apparel and
general retail demand, and retail destocking rather than to a loss of interest; the market
remains roughly 50% larger than four years earlier. Volume projections for 2028 exceed 100
billion chips (Grade C: industry association data, self-reported by members).

Unit price has fallen to roughly EUR 0.04–0.05 per inlay in bulk, from around EUR 0.75 twenty
years ago (Grade C/D: vendor and association sources, quoted as ranges rather than list prices).
At four cents, tagging a low-value item becomes arithmetically possible; the binding cost moves
to applying the tag, which is labour.

### 9.2 Container telematics: what carriers actually publish

| Operator | Deployment | Reporting cadence | Stated coverage | Grade |
|---|---|---|---|---|
| Hapag-Lloyd | 1.6 million dry containers with IoT trackers | 15 min on land; ocean leg via API at 24 h | 97–98% cellular coverage claimed | C |
| Maersk | IoT connectivity upgrade on 450 vessels, completion targeted Q1 2026 | Device-dependent; reefer datalogs hourly for over 80% of units | Satellite absent on partner and short-charter vessels | C |

Maersk's own product documentation is unusually candid about gaps: reefer data can be delayed or
missing when the unit is off power, and there is no satellite coverage on partner vessels,
foreign feeders, and short-term charter vessels. That is a statement about the physical
installations, not about the software, and it is the kind of limitation a vendor page rarely
volunteers (Grade C).

### 9.3 RFID accuracy: claims against field evidence

| Source | Setting | Reported result | What the method omits | Grade |
|---|---|---|---|---|
| Bertolini et al., 2015, *International Journal of RF Technologies* | Italian fashion retailer, 35 RFID counts against barcode | 98–99.9% SKU accuracy; accuracy follows a logarithmic curve against counting time | Single retailer, favourable category | A |
| Jimenez-Chiang et al., 2025, IEOM Asia Pacific | Peruvian denim manufacturer, simulation plus pilot | Inventory record accuracy 85% to 95%; registration time down 81% | Small firm, modelled benefit | A |
| Echeverri-Arias et al., 2025, *JISEM* | 18 Colombian fashion stores, 2018–2021 | Claimed above 99% inventory reliability | Self-reported; read environment not described | A/C |
| Shapoval, 2025, *IJSR Archive* | Retail chain pilot, statistical audit method | Accuracy 72% to 99.4% | Pilot scale; method is the contribution | A |

The pattern across two decades is consistent: RFID improves inventory accuracy substantially,
and the improvement follows diminishing returns against counting effort. The 98–99.9% figures
are achievable *within a defined read environment*. They do not survive an audit that counts
units outside that environment, which is why the retailer in section 1 measured 91.4%.

### 9.4 Vision pilots and the false-positive gap

| Claim | Source | What can be verified | Grade |
|---|---|---|---|
| 10,000+ truck detections at 99% accuracy for plate and DOT number reads | Ryder and Terminal Industries pilot, reported by *Logistics Management* | Pilot scope; accuracy definition not published | C |
| Parcel damage classifier at 98.5% accuracy, false positives cut from 41 to 23 | AGH University, FedCSIS 2025 | Peer-reviewed, single dataset, warehouse lighting conditions | A |
| Tampering detection at 0.81 accuracy, 0.91 precision | TAMPAR, WACV 2024 | Peer-reviewed; high-precision operating point chosen deliberately | A |

The peer-reviewed literature clusters precision between 0.91 and 0.97 for damage and tampering
detection at high-recall operating points. That is a respectable result and it is not the same
as "solves damage detection", because the operating point is chosen against a cost ratio that
the paper does not know. Section 12 prices that choice.

### 9.5 Data volume and edge economics

A 2025 study of condition monitoring measured a single induction motor at 50 kHz sampling
producing approximately 34.5 GB per 24 hours raw, against roughly 830 MB per 24 hours after
feature extraction, a reduction of more than 40 times. The same study found a cloud instance
costing about USD 0.80 per hour against a USD 35 edge device, with the cloud faster at training
and the edge cheaper for continuous inference (Grade A for the measurement, in a controlled
laboratory setting).

A separate 2025 modelling paper puts a typical IoT device at 2.4 GB per day and estimates
centralised cloud cost at roughly USD 263 per device per year, falling to about USD 66 with 80%
of processing at the edge (Grade C: modelled, not measured). Treat the second set of numbers as
an order of magnitude, not a quote.

The design rule that survives both sources: **transmit facts, not samples.** A device that sends
an excursion flag and a summary statistic costs a thousandth of a device that streams raw
readings, and is usually sufficient for the decision. Raw streaming is justified when you do not
yet know which question you will ask, which is a research condition, not an operating one.

### 9.6 What the evidence does not support

- No public source quantifies false-positive rates for trailer counting in production yards. The
  figures that circulate are vendor claims.
- No peer-reviewed study establishes a general inventory-accuracy figure for RFID across
  categories. The evidence is site-specific.
- Digital twin claims are ahead of deployment. A 2025 systematic review of 31 supply chain
  digital twin papers found that only about 13% combined visibility, optimisation, prediction,
  and simulation, and identified data accuracy and implementation cost as unresolved gaps. A
  separate 2025 review of warehouse digital twins concluded that most remain experimental, with
  no standard metrics.
- Cost-per-excursion figures for pharmaceutical shipments (commonly USD 3,000–10,000 for the
  investigation alone) come from vendor and consultancy pages, not audited data. They are
  usable as an order of magnitude and should be labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reading treated as an event | Alert volume far exceeds investigation capacity | No episode rule; single-sample triggers | Alert fatigue, then ignored alerts | Define threshold, duration, hysteresis, minimum samples |
| Logger placement unrepresentative | One logger disagrees with all others | Sensor in ambient air or against a panel | Wrong disposition in both directions | Place by mapping or validation study; record placement |
| Sampling interval too long | Short excursions never appear in the data | Aliasing; event shorter than interval | Understated exposure | Set interval against the shortest event to detect |
| Clock skew | Event timestamps inconsistent across devices | Real-time clock drift without sync | Wrong attribution to a leg or a custodian | Store UTC offset and sync status; check skew |
| Calibration expired | Certificate dates not in the record | Device in service past its interval | Evidence inadmissible in a dispute | Automate calibration status on ingest |
| Gap read as normal | Long series with no gaps in one lane only | Missingness treated as absence of condition | Understated excursions, false confidence | Classify gap causes; carry an unknown state |
| Read-rate claim taken as accuracy | Audit recounts find fewer units than the system | Claim measured in a favourable read zone | Inventory record wrong; service failures | Define the counted population; audit independently |
| False positives unpriced | Operators complain but no cost figure exists | Detection threshold chosen for recall alone | Labour cost exceeds benefit | Price both error types; choose the threshold on cost |
| Unit or scale error | Values plausible but biased | °F in a °C field; resolution too coarse | Systematic error invisible in summary statistics | Validate ranges and resolution on ingest |
| Counterparty owns the record | No raw export; portal only | Carrier or 3PL controls retention and edits | Cannot audit or dispute | Contract for raw export and immutable log |
| Aggregation hides a failing device | Fleet statistics normal, site statistics poor | Mean over devices masks one bad sensor | Localised quality failure undetected | Monitor per device, not per fleet |
| Duplicate reads | Dwell times implausibly long | Retrying gateway or overlapping readers | Inflated dwell, wrong detention charges | Deduplicate on device and message identifier |

Two deserve emphasis.

**Alert fatigue is a design failure, not a user failure.** The 412-alert quarter in section 1
was produced by a rule with a threshold and nothing else. Once the team stopped reading, the
system's effective sensitivity became zero, which is worse than no system because it carries a
licence cost and a false sense of coverage.

**Counterparty ownership is a contract problem that becomes a data problem.** If the carrier's
portal is the only source, you cannot recompute a statistic, cannot audit an edit, and cannot
retain beyond the portal's window. The time to fix this is at contract signature, which is
Topic 07's subject.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The device identity in the record is the device that measured | Every downstream join depends on it | Reused identifiers, cloned configurations, swapped devices | Compare serial numbers against the packing record |
| Calibration remains valid for the record's whole life | Accuracy claims rest on it | Device in service past its interval; certificate lost | Automate status checks at ingest and at review |
| The sampling interval is shorter than the shortest event that matters | Detection probability depends on it | Short excursions, door openings, shock events | Compute implied detection probability per event type |
| Missing data means "not covered", never "normal" | Prevents false confidence | Pipeline defaults that fill gaps with the last value | Count gaps by cause; assert an unknown state |
| The sensor location represents the quantity of interest | All inference assumes it | Placement chosen for convenience | Validation study comparing locations |
| Timestamps are comparable across devices and systems | Sequencing and attribution require it | Clock drift, timezone errors, daylight saving transitions | Measure inter-device skew on overlapping windows |
| The derived fact can be recomputed from retained data | Auditability requires it | Only aggregates retained; raw payload discarded | Recompute a sample of episodes from raw and compare |
| The read or detection environment is stable | Accuracy claims are conditional | Rack layout change, new lighting, new product mix | Re-baseline after any physical change |
| The counterparty will supply raw records on request | Disputes depend on it | Portal-only access; contractual restriction | Request a raw export before signing, not after |
| False-positive cost is stable per event | Threshold optimisation uses it | Queue saturation makes each event more expensive | Measure clearance time at realistic volumes |

Three limits of the framing are worth stating plainly.

**The framing is about measurement, not about prediction.** Nothing in this topic tells you what
a shipment will do next. Sensing gives you a better estimate of the current state; forecasting
that state is a different discipline, treated in Modules 06 to 20.

**Cost figures in this topic are order-of-magnitude.** Tag prices, telematics fees, and
investigation costs vary by a factor of five across regions and volumes. Use them to structure a
decision, not to populate a business case without a quote.

**Regulatory content is educational.** The instruments cited here change, and the applicable
version depends on jurisdiction and product. Confirm against the current consolidated text and
take professional advice before relying on any of it operationally.

## 12. Exercises

### Exercise 1 — Build the evidence ladder (analysis, 45 minutes)

Choose a sensing source you can actually access: a phone's location history, a smart meter, a
car's telemetry, or a public AIS feed.

**Deliverable:** the six-rung ladder for that source, with one sentence per rung naming what is
recorded and what is lost.
**Acceptance criteria:** all six rungs present; at least two rungs identify a specific loss that
cannot be recovered downstream; one rung names a party other than you who can alter the record.
**Time:** 45 minutes.

### Exercise 2 — Compute MKT (calculation, 50 minutes)

A 24-hour window contains 12 hourly readings at 4.0 °C, 8 at 6.0 °C, and 4 at 12.0 °C.

**Deliverable:** MKT to two decimal places, the arithmetic mean, and the share of the Arrhenius
sum contributed by the warm hours.
**Acceptance criteria:** kelvin used correctly; the arithmetic mean reported separately; a
one-sentence statement of what the gap between the two figures means commercially.
**Time:** 50 minutes.

### Exercise 3 — Sampling interval and aliasing (calculation, 40 minutes)

A logger samples every 15 minutes. Door openings last 9 minutes on average and occur at random
phase.

**Deliverable:** the probability that a given opening is detected at all, the expected number of
samples recorded per detected opening, and the interval you would specify to reach 95% detection.
**Acceptance criteria:** the probability stated as a ratio with reasoning; the required interval
derived, not asserted; one sentence on what the detected sample does and does not tell you.
**Time:** 40 minutes.

### Exercise 4 — Design an RFID read-rate test (design, 60 minutes)

You are asked to verify a vendor claim of 99% read accuracy for a pallet-level portal in a
warehouse holding metal fixtures and liquid goods.

**Deliverable:** a one-page test protocol.
**Acceptance criteria:** the counted population is defined; at least three product categories
are included, one metal-heavy and one liquid-heavy; the number of passes and the acceptance
threshold are stated; the protocol states what result would falsify the claim.
**Time:** 60 minutes.

### Exercise 5 — Price a detection threshold (calculation, 50 minutes)

A dock has 180 trailer movements per day and 300 operating days per year. Threshold A gives
recall 0.99 and precision 0.88; threshold B gives recall 0.95 and precision 0.985. A false
positive costs EUR 9.85 in clearance labour; a missed arrival costs EUR 65.

**Deliverable:** annual cost of each threshold, and the cost of a missed arrival at which the
two are equal.
**Acceptance criteria:** the false-positive count derived from precision and recall rather than
assumed; both annual costs in currency; the break-even cost stated.
**Time:** 50 minutes.

### Exercise 6 — Provenance memo (design, 45 minutes)

A carrier's portal is the only source of container temperature for a lane. Write a memo to
procurement specifying what the contract must require.

**Deliverable:** five contractual requirements with the operational reason for each.
**Acceptance criteria:** at least one requirement covers raw export; one covers tamper evidence
or audit trail; one covers retention period; each reason names a specific decision that fails
without it.
**Time:** 45 minutes.

### Exercise 7 — Size the data budget (calculation, 40 minutes)

A site has 40 sensors reporting a 16-byte payload every 10 seconds, plus 6 cameras at 2 Mbit/s
each.

**Deliverable:** daily and annual volume for sensors and cameras separately, and the fraction of
the total contributed by video.
**Acceptance criteria:** units converted correctly and shown; the comparison stated as a ratio;
one recommendation on what to process at the edge.
**Time:** 40 minutes.

### Exercise 8 — Disposition decision (design, 60 minutes)

Using the section 6 shipment, change one parameter: the contract limit is MKT ≤ 7.0 °C rather
than 8.0 °C.

**Deliverable:** the new verdict, the new disposition, the cost of each option, and the
probability threshold at which the stability test becomes worth buying.
**Acceptance criteria:** verdict recomputed rather than asserted; cost of information computed
against the value at stake; the recommendation states what happens to the remaining stability
budget.
**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
T(K): 4.0 -> 277.15, 6.0 -> 279.15, 12.0 -> 285.15
terms: 2.5164, 2.7683, 5.8827  (x 10^-16)

sum = 12(2.5164) + 8(2.7683) + 4(5.8827)
    = 30.1968 + 22.1464 + 23.5308 = 75.8740
mean = 75.8740 / 24 = 3.1614
MKT  = 10,000 / (-ln(3.1614e-16)) = 10,000 / 35.6906 = 280.19 K = 7.04 C
```

The arithmetic mean is 144 / 24 = 6.00 °C. The four hours at 12.0 °C are 17% of the time but
31% of the Arrhenius sum (23.53 of 75.87). A contract written as "MKT ≤ 7.0 °C" fails this
profile even though no reading exceeds 12 °C and the average is 6.0 °C; a contract written as
"mean ≤ 7.0 °C" passes it. The statistic, not the data, decides the outcome.

**Exercise 3 arithmetic.**

A 9-minute event against a 15-minute grid is detected when a sample instant falls inside it:
probability 9/15 = 0.60, and exactly one sample is recorded when it is caught. To reach 95%
detection the interval must satisfy `I ≤ 9 / 0.95 = 9.47 min`, so specify 9 minutes or, with
margin for clock jitter, 5 minutes. The single detected sample proves the door opened at some
point in a 15-minute interval; it does not establish when it opened, how long it stayed open, or
what the temperature did inside.

**Exercise 5 arithmetic.**

Threshold A: true positives 180 × 0.99 = 178.2; false positives 178.2 × 0.12 / 0.88 = 24.3;
misses 1.8. Daily cost 24.3 × 9.85 + 1.8 × 65 = 239.4 + 117.0 = EUR 356.4, or EUR 106,900 per
300-day year.

Threshold B: true positives 171.0; false positives 171.0 × 0.015 / 0.985 = 2.6; misses 9.0.
Daily cost 2.6 × 9.85 + 9.0 × 65 = 25.6 + 585.0 = EUR 610.7, or EUR 183,200 per year.

Break-even: 239.4 + 1.8c = 25.6 + 9c gives 7.2c = 213.8, so c = EUR 29.70. Below that cost per
missed arrival, the high-recall threshold wins; above it, the high-precision one does. The lesson
is that precision alone decides nothing: the same two models swap places when the cost ratio
moves by a factor of three.

**Exercise 7 arithmetic.**

Sensors: 40 × 16 B × 8,640 samples/day = 5,529,600 B = 5.53 MB/day, or 2.0 GB/year.
Cameras: 6 × 2 Mbit/s = 12 Mbit/s = 1.5 MB/s = 129,600 MB/day = 126.6 GiB/day, or 47.3 TB/year.
Video is 99.996% of the volume. Any architecture that treats "sensor data" as one problem has
misallocated its budget before it starts; the camera stream needs edge inference or it dominates
the network and storage bill.

A strong submission identifies a rung where the loss is permanent, for example a logger that
stores only the converted value so a later calibration correction cannot be applied, or a phone
location history that is thinned by the operating system before the application sees it. It also
names a third party who can alter the record: the platform vendor, the carrier, or the operating
system. A weak submission lists six technologies without naming a loss, which reproduces the
vendor diagram rather than analysing it.

**Exercise 4 rubric.**

A strong protocol defines the counted population as physical units verified by manual recount,
not units the system believes exist. It includes a metal-heavy and a liquid-heavy category,
specifies at least 30 passes per category, and states a falsification condition such as "if any
category falls below 95% on three consecutive passes, the claim is rejected for that category".
It also records tag placement, reader power, and antenna configuration, because a read rate
without those is not reproducible. A weak protocol tests one pallet of cardboard in an empty
bay and reports a single number.

**Exercise 6 rubric.**

Strong requirements are specific and testable: a daily raw export in a documented schema; an
append-only record with a hash or signature; retention for the longer of seven years or the
product shelf life plus one year; a documented device identifier and calibration status in every
record; and a right to audit the aggregation method. Each is tied to a decision that fails
without it, such as "we cannot recompute MKT under a different activation energy" or "we cannot
defend a rejection to the customer". A weak memo asks for "data access" and "good audit
practices", which no procurement team can enforce.

**Exercise 8 rubric.**

At a 7.0 °C limit the section 6 profile fails at 7.15 °C, so release without assessment is no
longer available. The assessment (EUR 4,200, 10 days) is the next step, and if it cannot
establish that the lot is within its stability data, the stability test at EUR 20,300 and 21 days
is justified whenever its probability of changing the disposition exceeds 1.91%. A strong answer
notes that the 15.8 degree-hours are already spent whichever route is taken, and that a
conditional release would consume the remaining budget with no reserve for the rest of the
product's life.

## 14. Interview questions

**1. "What is the difference between a sensor reading and a supply chain event?"**

A strong answer says the reading is a physical state at an instant and the event is a rule
applied to a series, with threshold, duration, and scope parameters that must be documented. A
weak answer treats them as the same thing with different formatting.

**2. "How would you set a sampling interval?"**

A strong answer starts from the shortest event that must be detected, computes the implied
detection probability, and then checks the battery and transmission budget. A weak answer says
"every minute" or "whatever the device defaults to".

**3. "What is MKT and why not use the average?"**

A strong answer gives the Arrhenius form, notes that degradation is exponential in temperature,
and states that MKT is always at least the arithmetic mean for a varying profile. A weak answer
says "it is a weighted average" without saying what the weights are.

**4. "A vendor claims 99% RFID accuracy. What do you ask?"**

A strong answer asks for the counted population, the read environment, the product categories,
the number of passes, and whether the figure is read rate or inventory accuracy. A weak answer
asks for a reference customer.

**5. "How do you decide whether a camera false-positive rate is acceptable?"**

A strong answer prices both error types, computes the annual cost at realistic volume, and
states the break-even ratio. A weak answer compares the precision figure to a rule of thumb like
95%.

**6. "Why does an AIS track have gaps, and what do you do about them?"**

A strong answer distinguishes terrestrial range, transponder status, and aggregator polling, and
says the gap must carry an unknown state rather than a default. A weak answer says "satellite
coverage is patchy" and stops.

**7. "What makes a sensor record admissible in a dispute?"**

A strong answer lists device identity, calibration status and traceability, clock source, raw
payload retention, and an append-only trail showing no post-hoc edits. A weak answer says "it is
logged in the system".

**8. "Where would you process video, at the edge or in the cloud?"**

A strong answer compares bandwidth cost and latency against model update frequency, and notes
that video dominates volume by orders of magnitude. A weak answer picks one without a criterion.

**9. "How would you distinguish a regulatory requirement from a contractual one in cold chain?"**

A strong answer says the regulation sets the method and the licence sets the numerical limit,
while the quality agreement sets the commercial consequence, and gives an example where the
contract is stricter. A weak answer treats GDP as the source of the 2–8 °C figure.

**11. "You inherit a monitoring system with 400 alerts a quarter. What do you do first?"**

A strong answer reconstructs the alerting rule, measures the distribution of episode durations,
and separates instrumentation faults from real events before changing any threshold. A weak
answer turns the threshold up.

## 15. Key terms

- **Telemetry** — automated measurement transmitted from a remote asset to a receiving system.
- **Sample** — one measurement at one instant, with a value and a UTC timestamp.
- **Sampling interval** — time between samples; it sets the shortest detectable event.
- **Calibration** — documented comparison against a traceable reference, with correction.
- **MKT** — mean kinetic temperature, an Arrhenius-weighted single equivalent temperature.
- **Activation energy** — Arrhenius parameter; 83.144 kJ/mol is the pharmacopoeial default.
- **Excursion** — a period outside the permitted band, defined by threshold and duration.
- **Stability budget** — cumulative tolerated exposure that excursions consume.
- **Cold chain** — controlled temperature sequence from manufacture to administration.
- **GDP** — Good Distribution Practice; sets method, not numerical limits.
- **Telematics** — telemetry from a vehicle or mobile asset, including position.
- **AIS** — Automatic Identification System; vessel identity and position broadcast.
- **Terrestrial versus satellite AIS** — shore reception within about 40 nautical miles versus
  orbital collection with per-pass latency.
- **Geofence** — geographic boundary used to infer an event from a position.
- **Dwell time** — time inside a geofence or at a facility, bounded by first and last read.
- **Reefer** — refrigerated container with an integrated refrigeration unit.
- **RAIN RFID** — passive UHF RFID in the 860–960 MHz band.
- **Read rate versus inventory accuracy** — tags read in a pass versus units correctly recorded.
- **RTLS** — real-time location system; **UWB** — ultra-wideband positioning radio.
- **False positive** — detection reported where no target exists; always has a labour cost.
- **Edge computing** — processing near the device; **store-and-forward** — local buffering.
- **Audit trail** — append-only, time-stamped record of creation, change, and deletion.
- **ALCOA+** — attributable, legible, contemporaneous, original, accurate, plus completeness,
  consistency, endurance, availability.
- **Digital twin** — model synchronised to a physical asset, with a measurable sync latency.
- **EPCIS** — GS1 event-data standard, with optional sensor elements in version 2.0.

## 16. Sources

**Peer-reviewed and academic (Grade A)**

- Bertolini, M., Bottani, E., Romagnoli, G., and Vignali, G. (2015). "The impact of RFID
  technologies on inventory accuracy in the apparel retailing: Evidence from the field."
  *International Journal of RF Technologies*. DOI: 10.3233/rft-150069.
- Catarinucci, L. et al. (2011). "Experimental Performance Evaluation of Passive UHF RFID Tags
  in Electromagnetically Critical Supply Chains." *J. Communications Software and Systems*.
  DOI: 10.24138/jcomss.v7i2.179.
- Jimenez-Chiang, A., Vera-Alonso, C., and Quiroz-Flores, J. (2025). "RFID-Enabled Inventory
  Accuracy Improvement in a Peruvian Textile Manufacturer." IEOM Asia Pacific.
  DOI: 10.46254/ap06.20250215.
- Echeverri-Arias, J. et al. (2025). "Design and Construction of an RFID-Based System for
  Increasing Inventory Reliability Levels." *JISEM*. DOI: 10.52783/jisem.v10i44s.8570.
- Naumann, A. et al. (2024). "TAMPAR: Visual Tampering Detection for Parcel Logistics."
  *WACV 2024*. Open-access proceedings.
- Walani, D. and Doorsamy, W. (2025). "Edge vs. Cloud: Empirical Insights into Data-Driven
  Condition Monitoring." *Big Data and Cognitive Computing*, 9(5), 121.
- Bouraya, S. and El Korchi, A. (2025). "Digital twin applications in supply chain management:
  A systematic literature review." *Uncertain Supply Chain Management*.
  DOI: 10.5267/j.uscm.2025.2.001.
- Sullivan, B. et al. (2023). "Total Cost of Ownership of Real-Time Locating System (RTLS)
  Technologies in Factories." *Procedia CIRP*. DOI: 10.1016/j.procir.2023.09.082.
- 2025 UWB warehouse positioning studies: *Computers* 14(12):510; *Applied Sciences* 15(5):2689;
  IEEE ICTC 2025, DOI: 10.1109/ictc66702.2025.11388598.

**Standards and regulatory (Grade B)**

- EU Guidelines on Good Distribution Practice, 2013/C 343/01, OJ C 343, 23 November 2013.
- EudraLex Volume 4, Annex 11 (computerised systems) and Annex 15 (qualification and validation).
- USP <1079>, <1079.2> (mean kinetic temperature), and <1079.3> (monitoring devices).
- IATA Temperature Control Regulations, current edition; IATA CEIV Pharma programme.
- WHO TRS 961, Annex 9.
- 21 CFR Part 11, in particular 11.10(b), (e), and (h); FDA *Part 11 Scope and Application*
  (2003); FDA *Data Integrity and Compliance with Drug CGMP* (2018); FDA guidance on electronic
  systems and records in clinical investigations (2024).
- GS1 EPCIS 2.0 and CBV 2.0, ref.gs1.org.
- ISO 6346 container identification; SOLAS V/19 AIS carriage requirements.

**Company cases and investor claims (Grade C)**

- RAIN Alliance, tag chip shipment figures for 2024 and 2025 (association data, April 2026);
  RFID Journal coverage of the 2024 figure.
- Hapag-Lloyd Container Live Position API documentation and HERE deployment note (2025).
- Maersk, IoT connectivity fleet upgrade (May 2025) and Captain Peter product documentation.
- Ryder and Terminal Industries yard vision pilot, reported by *Logistics Management*.
- GS1 US, *Retail's RFID Evolution* whitepaper (tag price and ROI figures).

**Vendor and analyst material (Grade D)**

- Vaisala, "Mean Kinetic Temperature in GxP Environments" application note.
- Eupry, temperature excursion guidance page.
- Roboflow yard management page; KoiReader KoiVision yard intelligence page; Kargo automated
  dock door scanning guide (2025).
- Semtech and WITRAC cold chain case study; Hubble Network reefer tracking guide; ORBCOMM
  PT 6000 datasheet; FreightMaths excursion cost calculator; World Courier "Cost of Failure".

**Press and anecdote (Grade E)**

- Practitioner blogs on excursion disposition cost (TrueCold, Mindlabs), used only for order of
  magnitude and labelled accordingly.
**Note on citations.** Vendor figures are reported as the vendor states them and dated in the
text. Where a number could not be traced to a primary source, the text gives a range or an order
of magnitude rather than false precision.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_reference-and-external.md`](04_reference-and-external.md)
- Next topic: [`06_data-movement.md`](06_data-movement.md)
- Related: [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md)
- Related: [`../01_supply_chain_foundations/08_transport-and-incoterms.md`](../01_supply_chain_foundations/08_transport-and-incoterms.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
