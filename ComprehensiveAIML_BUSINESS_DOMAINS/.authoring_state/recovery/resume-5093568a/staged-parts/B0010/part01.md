---
case_id: "CASE-18-02"
case_title: "Call-centre transcription quality for evidence-based objection reporting"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Speech analytics / call-centre quality"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "authorization-scope"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Customer-operations analytics owner"
source_index: "INDEX.md"
review_trigger: "Audio channel, rubric, language route, consent policy, or review-capacity change"
---

# Call-Centre Transcription Quality: Evidence Before Aggregation

**Evidence disclaimer.** Every count in this manuscript is a constructed teaching input, labelled
ILLUSTRATIVE at its point of use. No contact centre, telephony platform, ASR vendor, or regulator
result is asserted, inspected, or reproduced. The arithmetic is verified: each derivation is
recomputed in the text so a reader can redo it with a calculator. Where the text says a quantity
"would" or "is projected to", it is a design projection conditional on the stated assumptions, not
an observation. The only claims carrying external authority are the formula definitions, which
follow standard speech-recognition and queueing conventions cited in §14. Where evidence is absent
the text writes `unknown` or `not-measured` rather than inventing a number.

## 1. The One-Line Framing

Turn an authorised call recording plus its telephony metadata into evidence-linked transcript
spans, then decide per call whether to publish an objection count, route the call to a human
reviewer, or withhold it — because a fluent transcript is not proof that a customer objected.

The decision unit is not the call and not the transcript. It is the **objection span**: a bounded
interval of authorised audio, attributed to a speaker role, matching a written rubric, carrying a
replayable offset and a version manifest. Everything downstream — the aggregate count, its
denominator, its confidence interval, the reviewer queue that produced it — is a function of how
many such spans survive scrutiny.

Three outputs are admissible and they are not interchangeable:

1. **Aggregate candidate** — a span that passed every gate, with machine-checkable provenance, that
   may enter a published count under a pinned denominator.
2. **Route** — a span that a human must hear before anyone may count it, with the routing reason
   recorded, so the queue is auditable rather than a silent dumping ground.
3. **Abstention** — an explicit refusal (consent unclear, capture incomplete, role unresolved,
   language unsupported, rubric ambiguous) that removes the call from the numerator *and* the
   denominator, with the removal counted.

The output is never a performance score for an agent, never a customer risk decision, and never an
employment signal. That boundary is a design constraint, not a disclaimer bolted on at the end: it
is the reason the system refuses to emit per-agent aggregates at all.

## 2. Context & Background

### 2.1 What the workflow actually looks like

A contact centre is a chain of lossy handoffs, and every hop leaves a mark on the audio that later
shows up as an apparent model defect.

- **Capture.** A call arrives over SIP, is answered by an IVR or a queue, and is bridged to an
  agent. Recording may be trunk-side (before the PBX), station-side (at the handset), or
  screen-and-voice via the agent desktop. Trunk-side recording hears what the network delivered;
  station-side recording hears what the headset microphone produced. The two disagree about level,
  noise floor, and whether the customer's voice is attenuated by the agent's headset sidetone.
- **Codec.** Narrowband G.711 at 8 kHz with a nominal 300–3400 Hz passband is still common on
  PSTN-facing legs; wideband Opus at 16 or 48 kHz appears on softphone and WebRTC legs. The same
  sentence spoken by the same person has measurably different acoustic information depending on
  which leg carried it.
- **Loss and concealment.** Jitter buffers conceal lost packets by repeating the last frame. The
  result is not silence but a stretched vowel, which an acoustic model happily transcribes as a
  longer word.
- **Bridging and hold.** A consult transfer puts the customer on hold music, sometimes with a
  recorded announcement, sometimes with silence and comfort noise. Hold segments are speech-like
  to a voice-activity detector and contain no customer speech at all.
- **Dual-channel versus mixed.** Where the two parties arrive on separate channels, diarisation is
  nearly free and speaker attribution is a metadata lookup. Where they arrive mixed, attribution
  becomes an inference problem with its own error rate.
- **Consent and purpose.** Many jurisdictions require an announcement, a lawful basis, or both
  before a recording may be used for quality analysis beyond the original service purpose.
  Re-purposing a service recording for aggregate objection analytics is a distinct purpose in most
  privacy regimes, and the lawful basis is often narrower than the recording's existence suggests.
- **Retention.** Audio is typically retained for a short window (weeks to a few months) while
  derived text or features may be retained longer. Any claim that a report is auditable must
  survive the audio's deletion schedule; a span whose audio has expired is not evidence, it is a
  recollection with a timestamp.

### 2.2 Why the naive pipeline fails

The tempting pipeline is: transcribe everything, keyword-match objection phrases, count per day,
chart the trend. Each of the four steps has a specific failure that the naive design hides:

- **Transcribe everything** assumes every recording is authorised and complete. Neither is true for
  a whole day of traffic: consent varies by campaign, and captures fail (dropped bridge, storage
  timeout, late start).
- **Keyword-match** assumes the transcript is faithful at the exact words that matter. Entity-like
  tokens — product names, amounts, dates, references — are precisely where word error concentrates,
  and they are also where objection language lives.
- **Count per day** assumes the denominator is stable. It is not: the set of processed calls changes
  with capture success and consent coverage, so a rising count can be a rising numerator or a
  shrinking denominator.
- **Chart the trend** assumes the label means the same thing across time. A rubric edit, a model
  upgrade, or a new queue changes the meaning of the count without changing its name.

The trap this case is built around is the fourth one, because it is invisible in the data: two
months of counts plotted side by side look comparable even when the second month was produced by a
different rubric over a different route mix.

### 2.3 Who is in the room

Four roles with genuinely different objectives, and the design fails if any of them is modelled as
a rubber stamp:

| Role | Wants | Will push back on |
|---|---|---|
| Customer-operations analytics owner | A defensible number with an interval | A number whose denominator it cannot explain |
| Quality reviewer | Fewer, better-targeted items; replay that works first time | A queue that arrives stale and unreplayable |
| Privacy / compliance officer | Purpose limitation, minimisation, retention proof | Silent re-purposing of service recordings |
| Contact-centre operations lead | No agent-facing scorecard built from the output | Any per-agent slice escaping into workforce management |

The analytics owner is the decision owner for this case. The compliance officer holds a veto that
the design encodes as a hard gate rather than a review step, because a review step can be skipped
under deadline pressure and a gate cannot.

### 2.4 What this case is not

It is not a vendor comparison of ASR engines, not a claim about any model's accuracy, and not an
argument that automated objection detection should replace listening. It is an argument that the
*evidence contract* — what a span must carry before it may be counted — determines whether the
report survives an audit, and that the contract costs reviewer minutes that must be budgeted like
any other capacity constraint.

## 3. The Business Question

**Stakeholder question (ILLUSTRATIVE framing):** "Our quality team listens to a sample of calls and
files objection notes by hand. Can we produce a monthly aggregate of customer objections by
category, with each count traceable to a replayable moment in an authorised recording, without
turning the output into an agent scorecard and without breaking either our review capacity or our
consent commitments?"

Decomposed into the five decisions the system must actually make:

1. **Admissibility.** Is this recording usable for this purpose at all? Binary, per call, decided
   before any transcription compute is spent. Fails closed.
2. **Span candidacy.** Does this stretch of speech contain a rubric-matching objection? If yes,
   where exactly does it start and end, and which tokens carry the claim?
3. **Attribution.** Which speaker role said it — customer, agent, IVR, or unknown? A rubric-matching
   sentence spoken by the agent while reading a script is not a customer objection.
4. **Publication.** Does this span enter the aggregate as-is, go to a reviewer, or leave the count?
   This is where calibration and review capacity meet.
5. **Denominator.** Which calls are in scope for the report, and what happens to the ones removed
   for consent, capture failure, or unsupported language?

Decision options, stated so they can be compared rather than traded off rhetorically:

| Option | What it produces | Cost profile | Failure mode |
|---|---|---|---|
| A. Manual listening only | Reviewed notes from a sampling frame | High minutes per call, small sample | Coverage so thin that category trends are noise |
| B. Transcribe, count matches, publish | Fast aggregate | Low marginal cost | Transcript error becomes report error, silently |
| C. Transcribe, gate, route uncertainty, sample the rest | Aggregate with interval and audit trail | Low base cost plus bounded reviewer load | Reviewer queue becomes the bottleneck if unbudgeted |
| D. Transcribe, publish per-agent rates | Agent-level coaching input | Moderate | Out of scope by design; employment-adjacent use |

Option D is listed to be rejected explicitly. It is the option an operations lead will ask for,
and the only honest answer is that the sample sizes and the attribution error rate cannot support
per-agent inference even when the aggregate is sound.

Hard constraints, each of which can independently sink the project:

- **C-1 Consent and purpose.** No processing without a recorded lawful basis for this purpose.
- **C-2 No per-person output.** Aggregates only, above a minimum cell size, never joined to agent
  identifiers.
- **C-3 Evidence for every count.** A published number must resolve to spans, and every span must
  resolve to audio that still exists and can be replayed.
- **C-4 Explicit attribution uncertainty.** An unknown-role span cannot be counted as a customer
  objection; it must be routed or dropped, and the drop counted.
- **C-5 Capacity honesty.** The routing policy must be feasible at the peak hour, not merely on the
  daily average.
- **C-6 Version pinning.** Rubric, model, audio parser, and aggregation mapping move as one
  release bundle; mixed-version cohorts are reported separately or not at all.

## 4. Assumptions & Constraints

Every assumption below is a place where the design could be wrong in a way the data will not
announce. The check column names the artifact that would falsify it; the fallback names the
behaviour when the check fails. A design that cannot state its fallback is not ready.

| ID | Assumption | Check that can falsify it | Impact if false | Fallback |
|---|---|---|---|---|
| A-01 | The recording is authorised for aggregate quality analytics, not merely for service delivery | Consent/purpose record keyed to campaign and date | Processing is unlawful; every derived count is unusable | Deny the call; count the denial |
| A-02 | Call and leg identity are stable from capture to report | CTI event replay versus recording header | Spans attach to the wrong customer; denominator double-counts | Quarantine the call; exclude from both numerator and denominator |
| A-03 | Capture covers the full conversation, including transfers | Bridge-event timeline versus audio duration | Truncated calls look like calls with no objection | Exclude; never impute missing speech |
| A-04 | Gold annotation is independent of model output | Blinded double annotation with adjudication | Measured quality is inflated by anchoring | Re-annotate blinded; discard the anchored set |
| A-05 | Dual-channel audio exists for the majority of legs | Channel-count histogram per queue | Attribution error rises sharply on mixed legs | Route all mixed-channel spans |
| A-06 | Reviewer service time is roughly stationary across the queue mix | Time-stamped console events per disposition | Capacity plan understates load on complex spans | Re-estimate per stratum; staff to the worst stratum |
| A-07 | Rubric phrases are stable within a reporting month | Rubric version stamp per span | Month-over-month comparison is meaningless | Split the series at the version change |
| A-08 | Reviewers can replay the audio referenced by a span | Replay success rate on a sampled audit | "Evidence-linked" degenerates to "asserted" | Block publication for the affected cohort |
| A-09 | The audit sample is drawn from accepted spans, not routed ones | Sampling-frame log with inclusion probabilities | Accepted-error rate is invisible; precision is overstated | Resample from the full accepted frame |
| A-10 | Language coverage matches the traffic mix | Language identification versus queue configuration | Silent degradation on the uncovered tail | Abstain on unsupported languages; report coverage |
| A-11 | Audio retention outlives the reporting window | Retention schedule versus report cadence | Older counts become unauditable | Shorten the comparison window or freeze spans earlier |
| A-12 | Reviewer headcount is available at the peak hour | Roster overlap with the arrival curve | Queue explodes exactly when traffic peaks | Cap routing at the peak-feasible share |
| A-13 | Objection language is expressed in words, not only in prosody | Comparison of lexical matches to human-noted objections | Recall ceiling is set by phrasing, not by audio quality | Report "stated objections only"; route prosody-only candidates |
| A-14 | Aggregate cells are large enough to publish | Cell-size check per category and month | Small cells become de-facto individual disclosure | Suppress cells below the threshold; report suppression count |

Two structural constraints sit outside the assumption table because they are not falsifiable, they
are chosen:

- **No imputation of missing speech.** Where the capture is incomplete, the call is excluded. A
  model asked to "fill in" a dropped segment produces text that no reviewer can verify, which
  destroys the evidence chain that the whole design exists to protect.
- **No retro-fitted labels.** A span is labelled with the rubric version in force at review time,
  not the version current at report time. Re-labelling history to match today's rubric is how a
  trend line becomes fiction.

The cost of these constraints is real and should be stated up front: exclusion reduces the
denominator, and a reduced denominator makes every per-100-calls figure larger. §6 quantifies that
effect rather than waving it away, because a 7% exclusion and a 30% exclusion have very different
consequences for whether a monthly comparison is worth publishing.
