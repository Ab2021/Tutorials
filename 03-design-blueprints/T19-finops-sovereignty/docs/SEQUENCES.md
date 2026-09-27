# T19 — FinOps, Token Economics & Sovereignty: sequence diagrams

> `T19` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Production](../production/README.md) · [Case study](../../../01-case-studies/T19-finops-sovereignty.md)

Twelve sequences, one per decision in the design. Each is drawn so that the **failure is visible in the diagram rather than described after it** — which is the point, because three of this topic's measurements degrade without any metric reporting it.

---

## 1. One request through all four planes

The corpus's cost ladder applied to a single call, with the number each step actually contributes.

```mermaid
sequenceDiagram
    autonumber
    participant C as client
    participant GW as gateway
    participant PS as prefix store
    participant R as router (tier)
    participant LLM as model
    participant M as metering

    C->>GW: request (system prompt 61k, tools, retrieved 8k, question)
    Note over GW: PLANE 1 — ATTRIBUTION<br/>tags: team, feature, tenant, route, env
    GW->>PS: prefix lookup (system prompt + tool defs)
    alt prefix stable and seen before
        PS-->>GW: HIT — 61,200 tokens at 0.10x
        Note over PS: the read/write ratio for THIS<br/>prefix decides whether this was<br/>correct. Break-even = 1.39 reads.
    else prefix changed
        PS-->>GW: MISS — full price, plus a 1.25x WRITE
        Note over PS: a nonce or a reordered tool<br/>list invalidates everything downstream [T]
    end
    GW->>R: residency FILTER, then difficulty tier
    Note over R: out-of-region options REMOVED<br/>from the candidate set, not scored. [D]
    R-->>GW: tier = mid
    GW->>LLM: prompt → mid model
    LLM-->>GW: 300 visible tokens + 2,000 reasoning tokens
    Note over LLM: reasoning is billed at the OUTPUT<br/>rate and is absent from the response.
    GW->>M: cost event (schema t19.cost_event.v1)
    Note over M: 4 load-bearing fields:<br/>route.filtered_out · output_reasoning<br/>unit · sovereignty.cost_uplift_applied
    M-->>C: response
```

| Step | Reads | Banked | Note |
|---|---|---|---|
| prefix HIT | 61,200 tokens at 0.10× | **23.2%** of the bill at a 90% hit share | capped at **26.7%** by the asymptote |
| prefix MISS | full price + 1.25× write | **−0.35 units** | the only step here that can raise the bill |
| residency filter | removes options | 0 | no error tolerance, so not a score |
| reasoning tokens | 2,300 billed for 300 visible | — | **7.67×** amplification |

**The diagram's point:** four of the five cost decisions on this path are made *before* the model runs, and the fifth (reasoning) is invisible in the response. A cost dashboard built from model outputs alone sees one of them.

---

## 2. The discount curve, and the number that ends it

Why a better cache discount stops helping, drawn as a sequence of discounts rather than a table.

```mermaid
sequenceDiagram
    autonumber
    participant D as discount
    participant L as prefix line
    participant B as the bill

    D->>L: 2x
    L->>B: line 0.8600 → 0.5500 ; banked 11.2%
    D->>L: 5x
    L->>B: line 0.7760 → 0.2800 ; banked 19.8%
    D->>L: 10x  (the corpus's figure [T])
    L->>B: line 0.7480 → 0.1900 ; banked 23.2%
    D->>L: 50x
    L->>B: line 0.7256 → 0.1180 ; banked 26.0%
    D->>L: 100x
    L->>B: line 0.7228 → 0.1090 ; banked 26.4%
    D->>L: infinite (a FREE cache read)
    Note over L: line 0.7200 → 0.1000
    L->>B: banked 26.7%   <-- THE ASYMPTOTE
    B-->>D: "you cannot do better than this, at any price"

    Note over D,B: the uncached 10% of calls still pays full price,<br/>and nothing outside the prefix is touched at all.
```

**The diagram's point:** the return on the caching lever flattens by 10× and is finished by 50×. A business case that assumes a "better caching tier" will deliver more is assuming past the asymptote — and the asymptote is **26.7%**, which is 31.05% of bill times the 86.1% best-case line reduction.

---

## 3. The write premium — the only step that can be negative

```mermaid
sequenceDiagram
    autonumber
    participant T as traffic
    participant PS as prefix store
    participant B as the bill

    T->>PS: prefix P, seen ONCE
    PS->>B: write 1.25x + read 1x/10 = 1.350
    B-->>T: uncached would have been 1.000 — this is +0.350, a LOSS
    Note over PS: break-even at 10x read discount,<br/>1.25x write price: n > 1.39 reads

    T->>PS: prefix P, seen TWICE
    PS->>B: write 1.25 + 2 x 0.10 = 1.450
    B-->>T: uncached 2.000 — saving +0.550. Worth it.

    T->>PS: prefix Q, changes every call (a timestamp at the front)
    PS->>B: write 1.25 + 0.10 = 1.350, EVERY TIME
    B-->>T: never read twice. A permanent 35% surcharge,<br/>with the dashboard reporting "caching: ON".
```

**The diagram's point:** caching is not a switch, it is a per-prefix decision with a measurable break-even. The operating rule that falls out is **enable per prefix, alert when the read/write ratio falls below 2**. A global switch cannot express that rule, which is why the prefix store must report per-prefix reads (HLD §7).

---

## 4. The lever ladder — claim order against banked order

```mermaid
sequenceDiagram
    autonumber
    participant C as the CORPUS's claim
    participant A as achievable()
    participant B as the bill

    C->>A: distillation: 40x
    A->>B: 0.15 of model_tier (20%) x 0.80 = 2.4%
    Note over A,B: #1 by claim, #8 by banked

    C->>A: reasoning_gating: 15x
    A->>B: 1.00 of reasoning_tokens (15%) x 0.60 = 9.0%
    Note over A,B: #2 by claim, #4 by banked

    C->>A: prompt_caching: 10x
    A->>B: 1.00 of system_prompt (31.05%) x 0.746 = 23.2%
    Note over A,B: #3 by claim, #1 by banked

    C->>A: batch_lane: ~50%
    A->>B: 1.00 of the whole bill x 0.20 x 0.50 = 10.0%
    Note over A,B: #7 by claim, #3 by banked

    Note over C,B: EIGHT OF TEN LEVERS MOVE POSITION.<br/>The ladder is a list of DISCOUNTS;<br/>a programme banks discount x share_of_bill.
```

**The diagram's point:** two orderings derived from one source disagree, and the disagreement is the finding. The corpus's ladder is not wrong — it is a list of prices, and the plan that follows from it starts somewhere other than the top of the list.

---

## 5. The pricing slot — one lever, and the conflict that pays for it

```mermaid
sequenceDiagram
    autonumber
    participant P as the plan
    participant RC as reserved_capacity
    participant BL as batch_lane
    participant B as the bill

    P->>RC: value decision FIRST: 0.45 x 0.35 = 15.75 pts
    RC->>B: applied. Bill reduced.
    P->>BL: batch_lane, 0.20 x 0.50 = 10.00 pts
    BL--xB: SKIPPED — conflicts with reserved_capacity
    Note over BL,B: one saving, not two. Both reprice<br/>the same tokens.
    B-->>P: 15.75 pts banked from the slot

    Note over P,B: THE OTHER ORDER (ease_first, the natural one):
    participant E as ease_first
    E->>BL: batch_lane is EASIER (ease 4 vs 2) — taken first
    BL->>B: 0.20 x 0.50 = 10.00 pts applied
    E->>RC: reserved_capacity now
    RC--xB: SKIPPED — conflicts with batch_lane
    Note over E,B: 10.00 pts banked instead of 15.75.<br/>5.75 pts lost to the ordering rule alone.<br/>(The full spread across three orderings: 4.15 pts.)
```

**The diagram's point:** there is exactly **one** pricing slot per traffic slice, and both orderings are defensible readings of the same instruction. The difference is not effort or risk — it is which lever got there first, which is a decision, not a default.

---

## 6. The unit — two rows, three signs

```mermaid
sequenceDiagram
    autonumber
    participant F as Feb
    participant A as Apr
    participant U as the unit
    participant B as the business

    F->>A: cost 0.30c → 0.16c ; lines 630 → 91 ; files 8.2 → 3.6
    A->>U: per ARTIFACT
    U->>B: 0.5333 = -46.7%  "the price halved"
    A->>U: per FILE
    U->>B: 1.2148 = +21.5%  "each change costs more"
    A->>U: per LINE of delivered work
    U->>B: 3.6923 = +269.2%  "the work costs nearly four times as much"
    Note over U,B: SAME TWO ROWS. THREE SIGNS.<br/>The deliverable shrank faster than the price<br/>(lines to 14.4% of Feb, cost to 53.3%).

    B-->>U: which one was the programme measured on?
    Note over U: whichever denominator the person<br/>who built the dashboard chose. [D]
```

**The diagram's point:** the unit is chosen before the arithmetic and decides its sign. The design answer is the **unit memo** (HLD §6): the denominator, its owner, and the companion metric, signed before work starts.

---

## 7. The workload mix — the average that is the wrong basis for an order

```mermaid
sequenceDiagram
    autonumber
    participant M as the mix
    participant AV as the average
    participant T as the tail
    participant B as the bill (next year)

    M->>AV: 1,000,000 calls, mean cost 0.004194
    Note over AV: "the average call is cheap"
    M->>T: the agentic population: 10,000 calls at 0.1224
    T->>AV: that is 29.2x the mean, 1.0% of volume, 29.2% of the bill
    Note over AV,T: an order derived from the MEAN optimises<br/>a population that is 1% of the traffic

    M->>T: grow the tail 4.3x (the scenario's own rate [D])
    T->>B: chat 70.8% → 36.1% ; agentic 29.2% → 63.9%
    B-->>AV: the tail is now the majority of the bill
    Note over AV,B: and the lever that fails on the long tail<br/>(distillation) is now the lever whose<br/>failure is most expensive.
```

**The diagram's point:** the average is not wrong, it is answering a different question — *what does a typical call cost* rather than *where is the money*. Both are legitimate; only one is a basis for an optimisation order.

---

## 8. The reasoning-token asymmetry

```mermaid
sequenceDiagram
    autonumber
    participant U as user
    participant LLM as model
    participant MET as "tokens per output" metric
    participant BIL as the invoice

    U->>LLM: "what is 17 x 23?"
    LLM->>LLM: thinks for 2,000 tokens
    Note over LLM: NOT visible to the user, NOT in the response
    LLM-->>U: "391"  (300 tokens)
    LLM->>MET: visible output = 300
    MET-->>U: "efficient: 300 tokens per output"
    LLM->>BIL: billed = 300 + 2,000 = 2,300 at the OUTPUT rate
    BIL-->>U: 7.67x the cost the metric implies
    Note over MET,BIL: the metric is not merely uninformative --<br/>it is SYSTEMATICALLY OPTIMISTIC, because the<br/>way to improve it is to move work out of view. [T]
```

**The diagram's point:** the corpus's instruction to stop using `tokens per output` **[T]** has a mechanical reason, and it is visible only when the invoice and the metric are drawn side by side. The remedy is a bill layer with its own lever (`reasoning_gating`, worth 9.0%) and a quality eval on the tasks gated **off**.

---

## 9. Sovereignty — the prerequisite cap, and the posture that hides it

```mermaid
sequenceDiagram
    autonumber
    participant V as vendor
    participant P as posture: vendor_tee_lease
    participant R as the prerequisite rule
    participant REV as the review

    V->>P: "confidential computing with cryptographic attestation"
    P->>REV: trust = 0.85, attested
    Note over REV: a TEE attestation IS cryptographic evidence --<br/>the strongest closer the corpus describes. [T]
    REV->>R: is trust <= control?
    R-->>REV: control = 0.50 (the silicon and the model are NOT yours)
    R->>P: trust := min(0.85, 0.50) = 0.50
    P-->>REV: effective 0.413, capped ['trust']
    Note over REV: the attestation proves a property of an<br/>environment you did not choose. It is a TRUE<br/>statement about somebody else's machine.
    REV-->>V: posture fails the floor test
```

**The diagram's point:** the attestation is not false — it is **about the wrong machine**. The prerequisite rule is the only thing that catches this, and it must be data (LLD §2.4) so that a reviewer can see the rule rather than trust the function.

---

## 10. Residency — a filter and a score, side by side

```mermaid
sequenceDiagram
    autonumber
    participant RQ as request
    participant SC as score-based router
    participant FL as FILTER router
    participant AU as the audit

    RQ->>SC: candidates: [in-region 0.82, out-of-region 0.79]
    Note over SC: noise sd = 1.0, gap = 1.0
    SC->>SC: picks by score
    SC-->>AU: 15.87% of the time, OUT OF REGION
    Note over AU: 1,586.6 leaks per 10,000 requests
    AU-->>SC: NON-COMPLIANT at any non-zero rate.<br/>A legal property has no error tolerance.

    RQ->>FL: candidates: [in-region]
    Note over FL: the out-of-region option is REMOVED,<br/>not penalised
    FL->>FL: picks by score among legal options only
    FL-->>AU: 0 leaks, at any noise level
    Note over AU,FL: identical latency, identical quality ranking.<br/>The difference is one line: the candidate set.
```

**The diagram's point:** the fix is not a better score, a wider gap, or a lower noise — it is **the candidate set**. A score-based implementation has a parameter for tolerance; a filter does not, and that absence is the design.

---

## 11. Continuity — the mean and the floor

```mermaid
sequenceDiagram
    autonumber
    participant A as architect
    participant L as the four layers
    participant MEAN as the reporting
    participant REV as the review

    A->>L: 2 accelerators, 3 model families, 2 engines, 1 cloud region
    L->>MEAN: 3 of 4 layers covered → mean 0.75
    MEAN-->>A: "good continuity posture"
    L->>REV: floor = 0 (cloud_region has ONE option)
    REV-->>A: SINGLE-VENDOR. A regulator or a vendor<br/>gets to choose the layer that fails.
    Note over A,REV: the mean is what a slide shows;<br/>the floor is what an adversary gets to choose.

    A->>L: alternative: 1 accelerator, 4 families, 2 engines, 3 regions
    L->>MEAN: same mean 0.75
    L->>REV: floor = 0, binding layer = ACCELERATOR
    Note over REV: same reading, different binding layer.<br/>Which one is fixable? The engine, not the silicon.<br/>Accelerator choice is fixed EARLIEST. [T]
```

**The diagram's point:** the floor names the binding layer, and the binding layer decides the cost of the fix. Three of the four examples here score a mean of 0.75 and a floor of zero.

---

## 12. Three silent failures, in one diagram

The three measurements in this topic that degrade without any metric reporting it.

```mermaid
sequenceDiagram
    autonumber
    participant P as the programme
    participant U as the UNIT
    participant C as COVERAGE
    participant T as the TRUST CAP
    participant D as the dashboard
    participant B as the business

    P->>U: choose a denominator: cost per request
    U->>D: cost per request: FALLING
    D-->>P: green
    U->>B: cost per delivered outcome: RISING 3.69x
    Note over U,B: the metric that would have shown it<br/>was never computed. SILENT.

    P->>C: 90.46% coverage, target 90% → pass
    C->>D: coverage: PASSING
    D-->>P: green
    C->>C: one growth cycle: 90.46% → 82.43%
    Note over C: no threshold was crossed -- it drifted.<br/>Nobody tagged worse. SILENT.

    P->>T: posture table: all four dimensions scored
    T->>D: posture: SATISFIED
    D-->>P: green
    T->>T: vendor_tee_lease claims 0.85 trust, evidences 0.50
    Note over T: the cap is in the RULE, not the data.<br/>A review that reads the claims passes. SILENT.

    Note over U,T: all three fail the same way: each is a comparison<br/>against a quantity that is NOT on the dashboard.<br/>There is no threshold to alert on.
```

**The diagram's point:** these are not three bugs. They are one design property — **a measurement whose reference value lives outside the system it measures** — appearing in the unit, the coverage figure and the posture table. Each has the same remedy shape: compute the companion quantity (the per-outcome cost, the growth rate, the prerequisite) and put it *beside* the number rather than replacing it.

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | the ladder and its ordering instruction; "exhaust the free wins"; "roughly a tenth of the cost"; cache reads ~10× cheaper; the stable-prefix caveat; "optimization without measurement is just guessing" |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | the Feb→Apr artifact study; "tokens per output" as the metric to stop using; attestation; the three-way trust problem |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt` | the four dimensions; the system-property framing; the TEE trade-off; accelerator choice as implementation strategy |
| `Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt` | the 10–100× agentic compute multiplier behind the tail population in sequences 7 and 9 |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` | the eight-layer decomposition; the 69%/28% anchor; provider caching discount ranges; batch and provisioned pricing; the attribution-before-everything rule |
| `ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` | cascade tiering; distillation's long-tail failure; the reasoning multiplier range |

Related blueprints: [T14 routing & gateways](../../T14-routing-gateways/HLD.md) ·
[T15 autoscaling & SLO](../../T15-autoscaling-slo/HLD.md) · [T17 observability & evals](../../T17-observability-evals/HLD.md) ·
[T18 guardrails & security](../../T18-guardrails-security/HLD.md)
