# Sequences: Constrained Generation

> `T03` · [HLD](../HLD.md) · [LLD](../LLD.md) · [Case study](../../../01-case-studies/T03-constrained-generation.md)
> **Transcript coverage:** primary · [Cheat sheet](../../../00-cheat-sheets/T03-constrained-generation.md) · [Interview bank](../../../02-interview-questions/T03-constrained-generation.md) · [Runnable core](../run.py)

End-to-end flows for the constraint plane. Each step names where it can fail; the failure taxonomy
is in [LLD §7](../LLD.md#7-error-handling), and the degradation ladder is in
[HLD §9](../HLD.md#9-failure-domains--degradation).

---

## 1. Cold start — first request after a schema publish

```mermaid
sequenceDiagram
    autonumber
    participant AU as Schema Author
    participant REG as Schema Registry
    participant COMP as Grammar Compiler
    participant AUD as Audit
    participant CACHE as Grammar Cache
    participant GW as Gateway

    AU->>REG: publish schema v14 (customer 2718)
    REG->>COMP: compile(schema, v14, compiler=3.2.1)
    COMP->>COMP: parse -> layout -> link
    COMP->>AUD: audit(grammar, vocab)
    alt no dead ends and accept reachable
        AUD-->>COMP: ok
        COMP-->>REG: artifact (content-addressed)
        REG->>CACHE: insert key=(schema_id, v14, compiler 3.2.1)
        REG-->>AU: published
    else dead end or stranded state
        AUD-->>COMP: dead_ends=[7]
        COMP-->>REG: GrammarError
        REG-->>AU: rejected -- state 7 has no legal token
        Note over REG: The artifact is never published.<br/>No request ever saw it.
    end
    GW->>CACHE: lookup (customer 2718, v14)
    CACHE-->>GW: artifact
```

**Failure point:** a schema that compiles but fails the audit. `spec.audit.onFailure: reject_artifact`
is the only sane setting — publishing a grammar with a dead-end state means every request that
reaches that state produces a zero-mass step at runtime, which is a production incident instead of a
build failure. The `strict: false` switch exists so a schema author can compile a work-in-progress
locally; nothing non-strict is publishable.

**Second failure point:** the compiler version. The artifact's cache key includes it, so a compiler
bug fix invalidates every cached grammar rather than leaving stale artifacts serving traffic —
without it, a fix ships and nothing changes.

---

## 2. The normal path — masked extraction

```mermaid
sequenceDiagram
    autonumber
    participant DOC as Document
    participant R as Router
    participant MASK as Logit Mask
    participant ENG as Engine
    participant VAL as Validator
    participant ERP as Customer ERP

    DOC->>R: invoice (customer 2718)
    R->>R: surface=invoice-extraction, maxDepth=1 -> schema-regular
    R->>MASK: bind grammar v14, state=0
    loop every decode step
        MASK->>MASK: legal = dfa.legal_symbols(state)
        MASK->>ENG: logits with -inf outside legal
        ENG->>ENG: softmax, renormalise
        ENG->>MASK: sampled token
        MASK->>MASK: state = step(state, token)
        Note over MASK: legal set is a handful of tokens<br/>against a 128k vocabulary
    end
    MASK->>VAL: record + grammar version + policy version
    VAL->>VAL: parse against the schema version IN FORCE
    VAL-->>ERP: accepted
```

**Failure point:** a step whose legal set is empty. `assertSurvivorsPerStep: true` turns it into a
counted `GRAMMAR_FAULT` and a page rather than a silent zero-mass sample. It should be unreachable in
a healthy system, which is exactly why it pages when it happens.

---

## 3. A recursive schema hits the depth cap

```mermaid
sequenceDiagram
    autonumber
    participant R as Router
    participant PDA as Pushdown Compiler
    participant ENG as Engine
    participant VAL as Validator
    participant DLQ as Dead-letter

    R->>PDA: customs declaration, recursive schema
    PDA-->>R: counter machine, maxStackDepth=32
    loop line items
        ENG->>PDA: push on '{'
        alt depth <= 32
            PDA-->>ENG: legal set
        else depth would exceed 32
            PDA-->>ENG: DEPTH_EXCEEDED
            ENG->>R: over-deep at depth 33
            R->>R: onDepthExceeded = fallback_to_flat
            R->>ENG: rebind to the flat grammar for this schema
            Note over R: Counted as a weaker guarantee.<br/>Structure still enforced,<br/>nesting beyond 32 unbounded.
        end
    end
    ENG->>VAL: record (route=flattened)
    VAL-->>DLQ: nesting beyond the flat grammar's reach
```

**Failure point:** the cap is checked at push time, not at overflow. Checking at overflow means the
stack has already grown past its bound before anything notices, which under paged attention is a
memory-growth path rather than an error path. `onDepthExceeded: fallback_to_flat` is a real
weakening — the flat grammar enforces the first 32 levels structurally and the rest not at all — so
the route is stamped on the record and the validator is the backstop.

---

## 4. Empty candidate set — compile-time catch against runtime symptom

```mermaid
sequenceDiagram
    autonumber
    participant AU as Author
    participant C as Compiler
    participant A as Audit
    participant RT as Runtime

    Note over AU,RT: The fixture: a field whose value list is empty.
    AU->>C: compile(schema with an empty value list)
    C->>A: audit(grammar, vocab)
    A->>A: forward reachability -> dead ends
    A->>A: reverse closure from accept -> stranded
    A-->>C: dead_ends=[7], stranded=[0..7], ok=False
    C-->>AU: REJECTED at build
    Note over RT: The request that would have hit state 7<br/>never gets built. No incident.

    AU->>RT: (had the audit not run) decode reaches state 7
    RT->>RT: legal = [], survivors = 0, max prob = 0.0000
    Note over RT: A silent zero-mass step.<br/>softmax returns zeros, not NaN,<br/>so nothing throws -- it just stops.
```

**Failure point:** this is the sequence the whole audit exists to prevent. `softmax` returning zeros
for an all-`-inf` row is the right *arithmetic* contract — it stops one bad step poisoning a batch
with `NaN` — but it is the wrong *detection* contract, because a zero-mass step raises nothing. The
detection belongs at compile time, and the runtime assertion is the second net.

---

## 5. Token healing at a URL boundary

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine
    participant GR as Grammar
    participant H as Healer
    participant T as Tokenizer
    participant V as Validator

    ENG->>GR: state requires ':' next
    GR-->>ENG: legal = {':'}
    ENG->>H: trigger_fired(prev=':', offenders)
    H->>H: offender_list, very_short, no_boundary_whitespace, prefix_of_another
    Note over H: Heuristic fires -- healing is gated, not always-on.
    H->>T: vocabulary
    T-->>H: candidates starting with ':' -> {':', ':/', '://'}
    H->>H: argmax probs -> '://'  (p=0.62 vs ':' p=0.02)
    H->>H: assert surface_preserved: '://'.startswith(':')
    H->>ENG: advance one step, three characters
    ENG->>V: record
    V->>V: surface form unchanged by the heal
```

**Failure point:** building the candidate set from the masked legal set instead of from the full
vocabulary. The mask's legal set is `{':'}` — every multi-character token was already excluded — so
healing that starts there is a no-op that still costs a recompute. The candidates must be re-derived
from the prefix over the whole vocabulary.

**Second failure point:** `changed == False`. The argmax of the healed set is sometimes the token the
mask would have picked anyway. That is one wasted forward pass, and the ratio of
`healing_changed_total` to `healing_trigger_total` is the measurement that decides whether the
trigger list is earning its cost.

---

## 6. A semantic constraint on the code surface

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine
    participant S as Semantic Scorer
    participant D as Discriminator
    participant V as Validator

    ENG->>ENG: step in the constrained span
    ENG->>S: top-k candidates (k=200)
    S->>D: score(candidate | prefix) for each
    D-->>S: P(constraint | prefix)
    S->>S: posterior = softmax(log P(token) + log P(constraint|prefix))
    alt retained mass >= minMass
        S-->>ENG: winner (possibly not the LM's argmax)
    else retained mass < minMass
        S->>S: widen k and retry
        Note over S: The truncation has cut into the body<br/>of the distribution. The truncated-out<br/>winner is no longer a tail case.
    end
    ENG->>V: record
```

**Failure point:** the truncation blind spot, which is not a bug and cannot be fixed. A candidate the
constraint would prefer but the language model ranked below `k` is unreachable, by construction —
that is what makes the method affordable. `semantic.minMass` does not remove the blind spot; it makes
it *detectable*, by distinguishing "the top 200 held 95% of the mass and the answer was in there"
from "the top 200 held 60% and the answer may not be".

**Second failure point:** the discriminator is a model, not a grammar. It is not guaranteed to
satisfy the constraint `[T]`, which is why the semantic layer degrades first and the validator is
never relaxed.

---

## 7. The validator rejects — version skew

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine
    participant VAL as Validator
    participant REG as Registry
    participant MET as Metrics

    ENG->>VAL: record, grammar_version=v13, policy=2026.09.4
    VAL->>REG: schema in force for customer 2718
    REG-->>VAL: v14
    VAL->>VAL: parse against v14
    VAL-->>MET: reject{reason=version_skew}
    VAL->>MET: validator_grammar_disagreement_total++
    Note over MET: grammar_fault_total stays at 0.<br/>This is NOT a contradiction --<br/>it is the design working.
    VAL->>ENG: reject -> repair loop, budget 2
    ENG->>VAL: v14-bound record
    VAL-->>ENG: accept
```

**Failure point:** the interpretation, not the code. A validator reject alongside a clean
`grammar_fault_total` looks like the two layers disagreeing about the language. They are answering
different questions: the mask guarantees membership in the *language*, the validator guarantees
membership in the *schema version currently in force*. A record generated under v13 and validated
under v14 is exactly the case the mask cannot see and the validator exists to catch.

**Second failure point:** reading a high `reject{reason=bad_value}` as skew. That reason means the
grammar and the validator disagree about the language itself, which is a compiler or validator bug,
not a timing artefact. The runbook separates them for this reason.

---

## 8. Repair exhausted — the record reaches a human

```mermaid
sequenceDiagram
    autonumber
    participant VAL as Validator
    participant REP as Repair Orchestrator
    participant ENG as Engine
    participant DLQ as Dead-letter
    participant H as Human Reviewer

    VAL->>REP: reject(record, reason=bad_value)
    REP->>ENG: regenerate with validation error appended, same grammar
    ENG->>VAL: attempt 2
    VAL->>REP: reject(record, reason=bad_value)
    REP->>ENG: attempt 3 (budget exhausted at 2 retries)
    VAL->>REP: reject
    REP->>DLQ: route with grammar_version, policy_version, all attempts
    DLQ->>H: review queue
    Note over H: 0.5% of 5M records ~= 25k/month<br/>~= 2 reviewers, against 30 for<br/>the unmasked baseline.
```

**Failure point:** an unbounded repair loop. A record that fails three times against the same
grammar is not a weak model — it is a schema and a template that disagree, or a document the schema
cannot represent. Retrying further burns tokens and hides the signal. `repair_attempts` going bimodal
at the cap is the metric that names this, and the fix is a schema or template change.

**Second failure point:** reading the DLQ as a model-quality metric. It is a *contract* metric. The
0.5% residual is not a masking failure — it is the class the mask cannot see at all: semantic errors,
stale schemas, upstream document problems. The correct response to a rising DLQ is to look at
`validator_decisions_total{reason}` before looking at the model.

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt` — the syntactic/semantic split; the JSON schema-to-state-machine construction; "add minus infinity before softmax"; the ~10-of-100,000 sparsity; the regular/context-free/Turing hierarchy; token healing; FUDGE and the top-200 truncation; the three failure modes of hard masking
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` — the 128k vocabulary and its long tail, which make the legal set sparse
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_5_A_and_Best_First_Search.txt` — the search alternative for end-verifiable constraints
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — per-step logit-processor cost against a decode step's weight reads

**All sequence structure is `[D]`.** Corpus facts (`[T]`) are attributed inline; the degradation
ladder and the failure taxonomy are in [LLD](../LLD.md).
