# T18 — Guardrails & Security: sequence diagrams

> `T18` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Production](../production/README.md) · [Case study](../../../01-case-studies/T18-guardrails-security.md)

Eight sequences, one per decision in the design. Each is drawn so that the **failure** is visible in the diagram rather than described after it — which is the point, because three of this topic's controls weaken without any metric reporting it.

---

## 1. A hostile request through the four rail positions

The corpus's walkthrough, with each rail's decision marked, and with the number each rail actually contributes.

```mermaid
sequenceDiagram
    autonumber
    participant M as member
    participant IR as input rails
    participant RA as retrieval
    participant RR as retrieval_rail
    participant CTX as context assembly
    participant LLM as model
    participant OV as output_validator
    participant G as capability gate

    M->>IR: "Summarise Q3 for me"
    Note over IR: pattern r=0.88 → clean<br/>classifier r=0.92 → clean
    IR->>RA: forward
    RA->>RR: 5 chunks retrieved
    Note over RR: 1 chunk flagged (structural)<br/>verdict cached for the other 4
    RR->>CTX: 4 chunks + 1 flagged, tags attached
    CTX->>LLM: context with trust levels as DATA
    Note over CTX,LLM: instruction/data separation is<br/>textual, not structural — the model<br/>CAN be persuaded [T]
    LLM->>LLM: follows an instruction<br/>embedded in a retrieved chunk
    LLM->>OV: draft answer with an out-of-band URL
    Note over OV: exfil marker → BLOCK<br/>covers ALL paths — this is why it<br/>is worth 70.2% alone
    OV->>G: (blocked; no tool call)
    G-->>M: refusal

    Note over IR,G: WHERE THE ATTACK WAS ACTUALLY CAUGHT:<br/>not at the input rails. They saw 1 of 4 paths.
```

| Position | Saw the payload? | Would have caught it? |
|---|---|---|
| input pattern | no | — |
| input classifier | no | — |
| retrieval rail | **yes** | yes, if `structural` |
| output validator | **yes** | yes, if the payload produces a detectable response |

**The diagram's point:** in this run the input rails are not weak, they are *absent from the traffic*. The corpus's sentence — teams skip the rail that matters [T] — is a coverage statement, and this is what coverage looks like at the sequence level.

### 1.1 The same request with the retrieval rail skipped

```mermaid
sequenceDiagram
    autonumber
    participant RA as retrieval
    participant CTX as context assembly
    participant LLM as model
    participant OV as output_validator
    participant G as capability gate
    participant T as tool

    RA->>CTX: 5 chunks, NO scan
    CTX->>LLM: hostile chunk in context, unlabelled
    LLM->>OV: draft with a benign-looking tool call
    Note over OV: nothing to detect —<br/>the response is clean, the CALL is not
    OV->>G: send_member_email(...)
    Note over G: session trust = 2 (read untrusted content)<br/>class requires 4 → DENY
    G-->>T: never executes
```

**The gate catches it and the detector never could.** Not because the gate is a better detector — it detects nothing — but because the question it asks has an answer that does not depend on recognising the payload.

---

## 2. The same payload, four paths

One attack class, four routes, four different outcomes. This is the measurement in §3.3 of the HLD drawn as four sequences.

| Path | Share of mix | Full-stack catch | Why |
|---|---|---|---|
| `direct` | 16.0% | **99.2%** | four rails positioned on it |
| `retrieved` | 53.0% | **84.8%** | retrieval rail + output validator only |
| `tool_output` | 17.0% | **71.8%** | same two, and the payload arrives later |
| `uploaded_doc` | 14.0% | **66.1%** | same two, and the upload may not be scanned at all |

```mermaid
flowchart TB
    P["payload:<br/>'ignore prior instructions…'"]
    P --> D["member types it"]
    P --> R["inside a retrieved doc"]
    P --> T["inside a tool's JSON"]
    P --> U["inside an uploaded file"]

    D --> D1["input pattern ✗<br/>input classifier ✓"]
    R --> R1["retrieval rail ✓<br/>output validator ✓"]
    T --> T1["retrieval rail ✓<br/>output validator ✓"]
    U --> U1["retrieval rail ✓<br/>output validator ✓"]

    D1 --> CATCH["caught 99.2%"]
    R1 --> CATCH2["caught 84.8%"]
    T1 --> CATCH3["caught 71.8%"]
    U1 --> CATCH4["caught 66.1%"]

    style D1 fill:#e6ffe6,stroke:#008800
    style U1 fill:#ffe6e6,stroke:#cc0000
```

**Read the diagram downwards, not across.** The mechanism is identical in all four columns. Only the *positioning* differs. A rail cannot be evaluated without knowing which column the traffic is in.

---

## 3. The tool call — the gate decides, and the audit event is written

```mermaid
sequenceDiagram
    autonumber
    participant LLM as model
    participant G as capability gate
    participant REG as tool registry
    participant ST as session history
    participant Q as approval queue
    participant T as tool
    participant A as audit log

    LLM->>G: disburse_funds(claim_id, amount)
    G->>REG: risk class?
    REG-->>G: irreversible, min_trust 5
    G->>ST: session_trust(history)
    Note over ST: min over ALL 11 steps = 2<br/>step 6 read untrusted content<br/>(window no longer contains it)
    ST-->>G: 2
    G->>G: 2 < 5 and class == irreversible → APPROVAL
    G->>Q: enqueue {tool, args, session, reason}
    G->>A: tool_decision{session_trust:2, min_required:5,<br/>outcome:approval, history_min_trust:2,<br/>queue_depth:6, model_version}
    Note over Q: queue depth is the gate's ACTUAL strength.<br/>At depth 38.5 the reviewer is a coin flip.
    Q->>T: on approval → execute
    Q->>A: approval_decided{decided_at, approver, outcome}
    T-->>A: execution result
```

### 3.1 The same call, gated on the window instead

```mermaid
sequenceDiagram
    autonumber
    participant LLM as model
    participant G as capability gate
    participant W as context WINDOW
    participant T as tool

    LLM->>G: disburse_funds(...)
    G->>W: session_trust(window)
    Note over W: step 6's hostile document<br/>scrolled out at step 9.<br/>Window min = 4.
    W-->>G: 4
    G->>T: 4 < 5, but window-gating<br/>and a partial-credit rule → ALLOW
    Note over T: THE WRITE EXECUTES
```

**Two lines of code apart, and the second one executes.** The LLD states `session_trust` as `min(history)` precisely because this is a one-line bug that integration tests pass by accident.

### 3.2 `write_low` while dirty — dry-run, not deny

```mermaid
sequenceDiagram
    autonumber
    participant LLM as model
    participant G as capability gate
    participant DR as dry-run executor
    participant A as audit log

    LLM->>G: draft_note(content)
    Note over G: class write_low, min_trust 3, session trust 2
    G->>DR: simulate
    DR->>A: intended_action{args, diff}
    DR-->>LLM: "would have written X (not executed)"
    Note over LLM: the agent's plan stays OBSERVABLE,<br/>which is what "a dry run mode lets you<br/>watch what the agent would have done" [T] buys
```

**Deny would have produced the same safety and destroyed the evidence.** Dry-run is why the gate is deployable on day one without blocking work.

---

## 4. The trust tag's journey — eight hops, and what fail-open does at the end

```mermaid
sequenceDiagram
    autonumber
    participant ING as ingestion
    participant P1 as pipeline (parse)
    participant P2 as pipeline (chunk)
    participant P3 as pipeline (embed)
    participant VS as vector store
    participant RET as retriever
    participant CE as context encoder
    participant MB as MODEL BOUNDARY
    participant G as gate

    ING->>P1: content + {trust: retrieved_untrusted}
    P1->>P2: content + tag
    Note over P2: a parser that rebuilds the object<br/>drops unknown attributes
    P2->>P3: content + TAG LOST
    P3->>VS: content, no tag
    VS->>RET: content, no tag
    RET->>CE: content, no tag
    CE->>MB: content, no tag
    Note over MB: tag_missing posture = fail_open → privileged<br/>(or fail_closed → untrusted)
    MB->>G: session trust = 4 (fail-open)
    Note over G: gate 0.0450 → 0.0930<br/>2.07x weaker<br/>catch rate: 82.3%, UNCHANGED
```

### 4.1 The two postures, side by side

| Posture | Tag survival @3 hops, 3% loss | Privileged untrusted content | Legitimate content restricted | Gate failure |
|---|---|---|---|---|
| **fail-open** | 91.3% | **8.7%** | 0.0% | **0.0930** |
| **fail-closed** | 91.3% | **0.0%** | 7.9% | 0.0450 |

**The detection metrics are byte-identical across the two rows.** The only signal that distinguishes them is a per-hop tag-loss metric, and the only place it can be measured is at the boundary — because by then the tag is already gone.

```mermaid
flowchart LR
    A["tag present<br/>at ingestion 100%"] --> B["after parse"]
    B --> C["after chunk"]
    C --> D["after embed"]
    D --> E["after retrieve"]
    E --> F["at the model boundary<br/><b>assert here</b>"]
    F -->|"missing"| G["fail closed → 7.9% of<br/>legit traffic restricted"]
    F -->|"missing"| H["fail open → 8.7% of<br/>hostile traffic privileged"]
    style H fill:#ffe6e6,stroke:#cc0000,stroke-width:2px
    style F fill:#fff3cd,stroke:#cc8800,stroke-width:2px
```

---

## 5. The false-refusal optimum — two businesses, one detector, opposite answers

The same cheap filter (AUC 0.802) on the same traffic, with two cost structures.

```mermaid
sequenceDiagram
    autonumber
    participant U as request
    participant D as cheap filter (θ)
    participant P as product owner

    Note over P: scenario = health_payer<br/>L=50,000 F=50 → L/F = 20.4
    P->>D: θ* = −1.90
    U->>D: benign request
    D-->>U: REFUSED (97.1% of benign traffic)
    Note over U: cost-optimal.<br/>unshippable.

    Note over P: scenario = marketing_bot<br/>L=500 F=40 → L/F = 0.3
    P->>D: θ* = +1.75
    U->>D: benign request
    D-->>U: allowed (fpr 4.0%)
    Note over U: catch falls to 29.1%.<br/>also cost-optimal.
```

| | health_payer | marketing_bot |
|---|---|---|
| L/F | 20.4 | 0.3 |
| θ* | −1.90 | +1.75 |
| catch | 99.9% | 29.1% |
| **fpr** | **97.1%** | **4.0%** |
| allow-all cost | 1000.00 | **10.00** |
| block-all cost | **49.00** | **39.20** |
| block-all worse? | **False** (20x saving) | **True** (3.9x worse) |

**The optimum moves by 3.6 standard deviations between the two, and the two disagree about which endpoint is safe.** The corpus's *"a system that blocks everything is safe and useless"* [T] is correct for one row of this table and wrong by 20x for the other.

### 5.1 Why one threshold is the wrong architecture

```mermaid
flowchart TB
    T["one threshold θ"] --> A["regulated business:<br/>θ* ≈ block everything"]
    T --> B["marketing business:<br/>θ* ≈ allow everything"]
    A --> C["the product team overrides it<br/>→ a policy chosen by discomfort,<br/>not by the ratio"]
    B --> C
    C --> D["the L/F ratio is now<br/>unwritten and untested"]
    D --> E["re-fix by ARCHITECTURE,<br/>not by a better threshold → §6"]
    style E fill:#e6ffe6,stroke:#008800
```

### 5.2 The segment skew inside a single global rate

```mermaid
sequenceDiagram
    autonumber
    participant OP as policy owner
    participant D as filter at θ = 1.645
    participant MJ as majority (70%)
    participant TS as terse (10%)
    participant DJ as domain_jargon (12%)
    participant DL as dialect_second_language (8%)

    OP->>D: "target a 5% false-refusal rate"
    D->>MJ: 5.0% refused (0.7x)
    D->>TS: 8.9% (1.3x)
    D->>DJ: 10.7% (1.6x)
    D->>DL: 14.8% (2.2x)
    Note over DL: the global 5% is an average over a<br/>population that is not uniform.<br/>The segment paying 2.2x is the one<br/>least able to route around it.
```

---

## 6. The cascade — 2.4% of traffic reaches the expensive detector

The architectural fix for §5.1, at an identical 2% false-refusal budget on the health payer.

```mermaid
sequenceDiagram
    autonumber
    participant R as request
    participant C as cheap filter<br/>AUC 0.802, cost 1
    participant P as precise check<br/>AUC 0.967, cost 20
    participant B as decision

    R->>C: score
    alt flagged (2.4% of ALL traffic)
        C->>P: escalate
        P-->>B: precise verdict
    else clean (97.6%)
        C-->>B: pass
    end
    Note over C,P: benign is refused only if BOTH flag it.<br/>an attack is caught if EITHER does.<br/>That is the OPPOSITE composition<br/>from the rail stack (which ORs to catch).
```

| Architecture | catch @ fpr 2% | detector cost | total |
|---|---|---|---|
| single cheap filter | 19.662% | 1.0 | 805.36 |
| precise detector alone | **70.755%** | 20.0 | 313.43 |
| **cascade** | **99.996%** | **1.5** | **2.39** |

**Two things to carry out of this diagram.** First, the cascade composes the detectors' *errors* — the same layering move the rail stack makes — while paying the expensive component on 2.4% of traffic. Second, **paying 20x for a better detector is not a substitute for putting it in the right place:** 70.755% against 99.996% at the same budget.

---

## 7. Red team on a schedule — the sawtooth and the trough

```mermaid
sequenceDiagram
    autonumber
    participant ATK as attackers
    participant LIB as in-library population
    participant NOV as novel population
    participant RT as red-team run
    participant MET as the prescribed metric

    loop every period
        ATK->>NOV: rotation 6% of the in-library population
        Note over LIB,NOV: between runs NOTHING moves<br/>the novel stock back
    end
    RT->>NOV: on the run date: discovery 75%
    NOV-->>LIB: converted to a rule / a training example
    NOV->>MET: novel-class catch
    Note over MET: 7.9% → 7.9% across 24 months.<br/>MOVES 0.0 POINTS. Cannot fire.
```

| Cadence | Mean | **Trough** | Sev mean | Sev trough | Exposure |
|---|---|---|---|---|---|
| continuous | 96.7% | **93.4%** | 95.9% | 91.4% | 0.0% |
| quarterly | 87.5% | 73.6% | 83.7% | 66.7% | 9.2% |
| semi-annual | 77.0% | 62.5% | 71.0% | 54.5% | 19.6% |
| annual | 63.1% | **45.5%** | 55.7% | 38.0% | 33.6% |
| never | 45.4% | **24.7%** | 38.8% | 20.3% | **51.2%** |

### 7.1 Solving for the schedule instead of choosing it

```mermaid
flowchart LR
    T["target trough"] --> Q{"largest gap whose<br/>trough stays above it"}
    Q -->|"0.90"| C1["continuous<br/>(93.4%)"]
    Q -->|"0.75"| C2["continuous<br/>(93.4%)"]
    Q -->|"0.70"| C3["quarterly<br/>(73.6%)"]
    Q -->|"0.60"| C4["semi-annual<br/>(62.5%)"]
    style C3 fill:#fff3cd,stroke:#cc8800
    style C4 fill:#fff3cd,stroke:#cc8800
```

### 7.2 Where this run disagrees with the rest of the knowledge base

```mermaid
flowchart TB
    S["24 months, cadence = never"] --> M["mean catch<br/>77.8% → 24.7%<br/>drop 53.1"]
    S --> V["severity-weighted<br/>71.6% → 20.3%<br/>drop 51.2"]
    S --> N["novel-class catch<br/>7.9% → 7.9%<br/>drop <b>0.0</b>"]
    M --> M2["not hiding anything —<br/>it falls 53.1 points.<br/>The DRIFT dominates the skew."]
    N --> N2["silent BY CONSTRUCTION.<br/>It starts at its floor and<br/>stays there, so it can never<br/>cross a threshold."]
    style N fill:#ffe6e6,stroke:#cc0000,stroke-width:2px
    style N2 fill:#ffe6e6,stroke:#cc0000
```

The average-hides-the-tail signature that T08, T09, T10 and T17 each reproduce **does not reproduce here**, and the run says so rather than manufacturing a fifth instance. The severity gap *narrows* (6.3% → 4.4%). What replaces it is a different failure: **the instrument aimed at the population that is entirely severity-5 is provably incapable of firing.** Alert instead on the count of injection *attempts* by source — a metric that measures the adversary rather than the classifier.

**And the mean's 53.1-point fall is a level, not a rate.** A red-team result reported without the date it was measured is the same class of error.

---

## 8. The approval queue — saturation, and the metric that says so

```mermaid
sequenceDiagram
    autonumber
    participant AG as agents
    participant Q as approval queue
    participant AP as approvers (2 × 40/h)
    participant MET as metrics

    Note over AG,AP: capacity = 80/hour over 8h
    loop quiet day (ρ = 0.10)
        AG->>Q: 8/hour
        Q->>AP: depth 0.0
        AP-->>MET: approver accuracy 0.950<br/>gate floor 0.0226
    end
    loop busy day (ρ = 0.95)
        AG->>Q: 76/hour
        Q->>AP: depth 18.0
        AP-->>MET: approver accuracy 0.703<br/>gate floor 0.1336  (5.9x weaker)
    end
    loop oversubscribed (ρ = 1.00)
        AG->>Q: > 80/hour
        Q->>AP: depth ∞
        AP-->>MET: approver accuracy 0.000<br/>gate floor 0.4500  (20x weaker)
        Note over MET: EVERY OTHER GUARDRAIL METRIC<br/>IS STILL GREEN
    end
```

| Queue depth | Approver accuracy | Gate failure |
|---|---|---|
| 0 | 0.950 | 0.0225 |
| 5 | 0.874 | 0.0567 |
| 20 | 0.681 | 0.1437 |
| **38.5** | **0.500** | — the analytic break-even |
| 60 | 0.349 | 0.2927 |
| 400 | 0.001 | **0.4495** |

### 8.1 The scaling wall, and the only exit

```mermaid
flowchart LR
    V["20,000 irreversible<br/>actions/day"] --> A["approval queue<br/>= 63 reviewers<br/>(31.5x today's 2)"]
    V --> U["undo path per action<br/>= the engineering time<br/>to build the compensating action"]
    A --> A2["headcount grows LINEARLY<br/>with agent volume"]
    U --> U2["irreversibility does not<br/>have to"]
    style A2 fill:#ffe6e6,stroke:#cc0000
    style U2 fill:#e6ffe6,stroke:#008800,stroke-width:2px
```

| Demand/day | Approvers needed | × today |
|---|---|---|
| 200 | 1 | 0.5x |
| 1,000 | 4 | 2.0x |
| 5,000 | 16 | 8.0x |
| **20,000** | **63** | **31.5x** |

**The metric that says the control stopped being real is the approval RATE, not the request rate.** A rising approval rate against a stable request rate means reviewers have stopped reading — and no detection metric, no coverage number and no configuration diff will show it.

---

## 9. The three silent failures, in one diagram

The through-line of this topic: **three of the six experiments end in the same place.**

```mermaid
flowchart TB
    subgraph S1["1. A trust tag is lost"]
        A1["a middleware hop drops<br/>an attribute"] --> A2["gate 0.0450 → 0.0930<br/>(2.07x weaker)"]
        A2 --> A3["catch rate: 82.3%<br/>UNCHANGED"]
    end
    subgraph S2["2. A rail decays"]
        B1["attackers rotate to<br/>novel techniques"] --> B2["trough 45.5% at annual<br/>vs 93.4% continuous"]
        B2 --> B3["the prescribed metric<br/>reports a LEVEL, not a rate"]
    end
    subgraph S3["3. An approver queue saturates"]
        C1["agent volume exceeds<br/>reviewer capacity"] --> C2["gate floor 0.0450 → 0.4495<br/>(20x weaker)"]
        C2 --> C3["approval rate is<br/>the only signal"]
    end
    A3 --> X["A CONTROL WEAKENED<br/>WITH NOTHING REPORTING IT"]
    B3 --> X
    C3 --> X
    X --> Y["not addressed by another classifier.<br/>Addressed by instrumenting the CONTROLS,<br/>not only the attacks."]
    style X fill:#ffe6e6,stroke:#cc0000,stroke-width:2px
    style Y fill:#e6ffe6,stroke:#008800,stroke-width:2px
```

| Control | How it weakens | What reports it |
|---|---|---|
| trust tag | a hop drops an attribute | **nothing** — every detection metric is flat |
| red-team rail | attackers rotate to novel techniques | **nothing** — the novel reading cannot fire |
| human approval | the queue deepens | **only the approval rate**, which nobody watches |

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Guardrails_Stop_Injection_Leaks_Hallucination_NeMo_Guardrails.txt` | the rail positions and the hostile-request walkthrough (§1); the tool-call rail recipe (allowlist, schema, approval, dry run, audit) driving §3; "the biggest blast radius"; "a system that blocks everything is safe and useless" (§5); "a rail you tested in March may be bypassed by June" (§7); "red teaming has to be a recurring schedule, not a launch checkbox"; the metric list behind §7.2 and §8.1 |
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts_2/MCP_vs_A2A_How_AI_Agents_Connect_and_How_to_Govern_Them.txt` | "who is allowed to do what on whose behalf" and the access-policy gateway (§3); the gated-tool-call share metric (§8.1) |
| `Agentic_AI_Infra_transcripts_3/Gosia_Steinder_-_Beyond_Harnesses_Platform_Solutions_for_Agent_Reliability_Secur.txt` | the lack of instruction/data separation in the context as the source of novel agent threats (§1, §2); why zero-trust's a-priori interaction patterns do not hold for agents; the interception layer as a uniform enforcement point (§3) |
| `CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_10_Incorporating_Tools.txt` | the sandboxing ladder; the prompt-injection example behind §2; the JSON-escaping failure that makes the schema a security control |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | the model opening a network connection on load; attestation and the "is the model I am running the model I evaluated" question (§4) |
| `ai-system-design-guide-main/ai-system-design-guide-main/12-security-and-access/01-llm-security.md` | **"the trust level is data, not metadata"** (§4); the five-layer IPI defence and **"capability gating is the most underused defense"** (§3); the arms-race timeline behind §7.2; insecure output handling (§6) |
| `ai-system-design-guide-main/ai-system-design-guide-main/13-reliability-and-safety/01-guardrails.md` | action safety with risk classification (§3); structured-output validation with retry ceilings (§6); the guardrail metric list in §7.2 and §8.1 |

Related: [HLD](../HLD.md) · [LLD](../LLD.md) · [Production](../production/README.md) · [Runnable core](../run.py)
