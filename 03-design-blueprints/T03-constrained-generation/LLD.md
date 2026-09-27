# LLD: Constrained Generation

> `T03` · **Transcript coverage:** primary · [HLD](HLD.md) · [Sequences](docs/SEQUENCES.md) · [Case study](../../01-case-studies/T03-constrained-generation.md) · [Cheat sheet](../../00-cheat-sheets/T03-constrained-generation.md)

Provenance: `[T]` transcript, `[R]` repo/reference, `[D]` derivation by this author with assumptions shown.

This document describes the low-level design of the constraint plane: the data structures, the
per-step transition, the exact arithmetic of the mask, the healing rollback, and the accounting that
makes each mechanism's cost visible. Code in `sim/` is the executable version of the structures named
here; the production shapes in `production/` are reference-grade and were **not executed in this
environment**.

## 1. Module Map

```
sim/
  automata.py     schema -> DFA; bracket FSA vs counter PDA; independent validator
  masking.py      -inf before softmax; renormalise; survivors; grammar audit
  healing.py      rollback, prefix-constrained candidate set, trigger heuristics
  fudge.py        top-k truncation, reward-augmented posterior, contrastive subtraction
  experiments.py  one experiment per mechanism
run.py            eight sections, one per claim
```

The split is deliberate: `automata.py` knows nothing about probabilities, `masking.py` knows nothing
about where the automaton came from, and `healing.py` and `fudge.py` never touch either, so each
mechanism is separately falsifiable. Only `automata.py` changes for production, where the DFA becomes
a compiler service; the other three carry over unchanged.

**The validator lives in `automata.py` next to the compiler in the simulation, and in production it
does not.** `validate_schema` is a direct parser, not a walk over the compiled table, so a compiler
bug cannot make the validator agree with it; in production they are separate services with separate
schema sources.

## 2. Core Data Structures

**DFA.** `transitions: dict[int, dict[str, int]]`, `accept: set[int]`, `start: int`. A *missing*
symbol is an illegal transition — the absence is the information the mask is built from, so the
representation never stores an explicit "illegal" marker. `n_states` is `len(transitions)`, and
`live_states()` returns the set reachable from `start`.

**Compiled schema layout (all fields required).** Four states per field plus a start and an accept:

| State | Meaning | Formula |
|---|---|---|
| `0` | expect `{` | — |
| `K_i` | expect key `i` | `1 + 4i` |
| `A_i` | expect `:` | `2 + 4i` |
| `B_i` | expect a value for field `i` | `3 + 4i` |
| `C_i` | expect `,` or `}` | `4 + 4i` |
| `ACCEPT` | after `}`, no fields left | `1 + 4n` |

So a two-field schema compiles to 10 states with accept 9 — the count `exp_compile_and_mask` prints.
The layout is arithmetic rather than a graph search so the state identity is inspectable by hand;
the cost is that a schema with a syntax error produces a colliding layout rather than a compile
error, which is why the compile step ends with `audit_grammar` rather than trusting the arithmetic.

**Grammar artifact.** Content-addressed by `(schema_id, schema_version, compiler_version)`. The
compiler version is in the key so a bug fix invalidates cached grammars; without it a fix ships and
nothing changes.

**PDA stack.** A list of symbols per sequence. Depth is `len(stack)`; the depth cap is enforced at
push time, not at overflow, so an over-deep request falls back rather than crashing.

**Per-request mask state.** `{sequence_id: (grammar_id, automaton_state, stack, depth)}` held by the
engine's logit processor, indexed by sequence and never by slot: a slot is recycled between requests,
and a state carried on the slot would leak a grammar into the next request's first token.

**Candidate set at a step.** The surviving token ids after masking, as a sorted array of int32. This
is the object `healing.candidates_starting_with` re-derives a superset of.

## 3. Interfaces & Contracts

```
compile(schema: Schema, version: str) -> GrammarArtifact
    # Raises GrammarError on: unreachable accept, dead-end state, ambiguous layout.
    # Postcondition: audit(artifact).ok is True.

audit(grammar: GrammarArtifact, alphabet: Vocab) -> AuditResult
    # n_states, n_reachable, dead_ends, stranded, ok

legal_symbols(state, alphabet) -> list[str]        # the mask's input
step(state, symbol) -> state | None                # None == illegal

masked_logits(logits: list[float], allowed_ids: list[int]) -> list[float]
    # Every disallowed index becomes -inf. Length preserved.

softmax(logits) -> list[float]
    # Tolerates -inf entries and an all -inf row (returns all zeros).

heal(candidates, prefix, probs, vocab) -> HealResult
    # chosen, chosen_text, chosen_prob, naive, naive_text, naive_prob,
    # changed, surface_preserved, error

trigger_fired(prev_token_text, vocab, offender_list) -> list[str]
    # [] means no trigger fires; each element names a heuristic

reward_augmented(probs, discriminator, k) -> {ids, posterior, winner,
                                              unconstrained_winner,
                                              truncated_out}
```

Three contracts carry the design's safety properties, and each is stated as a postcondition a test
can read:

1. **`softmax` on an all-`-inf` row returns zeros, not `NaN`.** `masking.softmax` filters `-inf` out
   of the max before exponentiating, so a fully masked row gives `total == 0.0` and returns
   `[0.0] * n`. The naive `math.exp(v - m)` gives `exp(-inf - -inf) = nan` and poisons the step.
   This is why the empty candidate set is caught by the audit rather than the softmax: that
   contract is "do not propagate NaN", not "detect a bad grammar". The runtime symptom —
   `exp_empty_candidate_set`'s zero survivors and max probability `0.0000` — is a silent zero-mass
   step in a real engine, not an exception.
2. **`heal` sets `surface_preserved` by checking `vocab[chosen].startswith(prefix)`, not by
   construction.** It is a redundant check on the algorithm's own output, deliberately: it is the
   assertion that catches a candidate set built from the wrong prefix.
3. **`audit_grammar` returns both `dead_ends` and `stranded`.** Different bugs: a dead end cannot
   proceed; a stranded state can proceed forever and never terminate. Only the second runs to the
   length cap, and only the second is invisible to a step-by-step check.

## 4. State Machines

**Compile-time (the grammar's own).** `parse → layout → link → audit`. Failure in `audit` is a build
failure: the artifact is never published. The two audited properties are exactly the two that make a
mask well-defined — no reachable state without a legal token, every reachable state can still reach
accept.

**Request-time (per sequence).**

```
  INIT ──lookup grammar──> MASKED ──step ok────> MASKED   (loop)
                             │
                             ├── no legal token ──> GRAMMAR_FAULT ──> DLQ
                             ├── depth cap hit ──> FALLBACK
                             └── accept reached ──> HEALING ──> VALIDATE
                                                        │
                            VALIDATE ──invalid, budget──> MASKED
                                     ──invalid, exhausted──> DLQ
                                     ──valid──> EMIT
```

`GRAMMAR_FAULT` is unreachable in a healthy system because the audit ran at compile time. It exists
so an unexpected one is a counted event rather than a crash — dead-letter the request and page,
because the compiler is wrong.

**Healing sub-machine.** `DETECT(offender_list | very_short | no_boundary_whitespace |
prefix_of_another_token) → ROLLBACK(1) → REBUILD(candidates starting with prefix) → PICK(argmax) →
ASSERT(surface preserved) → ADVANCE`. The corpus allows "very rarely" two `[T]`; the production
config exposes that as `max_rollback`, default 1.

## 5. Algorithms

### 5.1 The mask

```
legal   = dfa.legal_symbols(state, VOCAB)          # O(|alphabet|)
allowed = {VID[t] for t in legal}
logits' = [v if i in allowed else -inf]
probs   = softmax(logits')                          # renormalise
```

The corpus states this exactly: "all the things that are allowed as the next token are zero, all the
things that are not allowed to the next token we add minus infinity and then we take the softmax
again to renormalize" `[T]`.

The interesting quantity is **how much mass the mask moves**, which `renormalisation_loss` computes
as `1 - sum(raw_probs[i] for i in allowed)`. Two regimes:

- *The model already agrees* — the legal set holds nearly all the mass, and masking is a no-op on
  the distribution. This is the common case in a fine-tuned extractor.
- *The model disagrees* — the mask deletes most of the mass and renormalises the remainder upward.
  The output is still valid, and it can be degenerate: the highest-probability legal token may be one
  the model would never have chosen. **This is the mechanism by which a bad grammar produces bad
  output rather than an error**, and it is why the mask is not a substitute for schema quality.

### 5.2 The grammar audit

```
reachable = live_states(start)
dead_ends = [s for s in reachable if s not in accept and legal_symbols(s) == []]
back      = reverse edges over reachable
can_accept = closure(back, accept)
stranded  = [s for s in reachable if s not in can_accept]
ok        = not dead_ends and not stranded
```

Both directions are needed. Forward reachability alone finds dead ends; the reverse closure from
accept is what finds a state that can emit tokens forever without ever terminating. The
`exp_empty_candidate_set` fixture is a schema whose second field has an empty value list: its value
state has no legal token, the audit reports `dead ends [7]`, and the runtime path in the same run
shows `survivors 0, max probability 0.0000`. That pairing — the audit naming the state, the runtime
showing the symptom — is what the runbook cites when it says the assertion must run at build time.

### 5.3 DFA vs PDA at the nesting boundary

A fixed depth *is* expressible: `bracket_fsa(max_depth)` builds `max_depth + 1` states where state
`k` means "k unmatched braces", with state 0 as both start and accept. It accepts `{`^n`}`^n for
`n ≤ max_depth` and rejects `n = max_depth + 1`. The state count is `d + 1`; the machine is correct
and it is bounded.

`CounterPDA` accepts the same language for *any* `n` with **1** state, because its memory is a stack
rather than a state set. The measured table:

| Declared depth | FSA states | PDA states | accepts ≤ d | accepts d+1 | PDA accepts d+1 |
|---|---|---|---|---|---|
| 3 | 4 | 1 | yes | NO | yes |
| 4 | 5 | 1 | yes | NO | yes |
| 8 | 9 | 1 | yes | NO | yes |

The pattern — the FSA is correct up to its declared depth and silently wrong past it, while the PDA
is unbounded — is the executable form of "anything that supports uh JSON schemas is actually writing
push down automa to enforce its constraints, not FSAs" `[T]`. **What the model does not show** is a
real production schema; it shows a one-symbol language where the boundary is trivially visible.

### 5.4 The combinatorial cost of flattening optionality

When fields are optional, the automaton must distinguish which *subset* remains, so the key state is
keyed by the remaining-field set. The measured state count:

| Optional fields | States | vs previous |
|---|---|---|
| 1 | 13 | — |
| 2 | 33 | 2.54x |
| 3 | 81 | 2.45x |
| 4 | 193 | 2.38x |
| 5 | 449 | 2.33x |

The ratio sits above 2 at these sizes and settles toward it, because the count is dominated by the
`2^n` remaining-subsets term plus the per-field states. Extrapolating `[D]`: a 20-field optional
schema is on the order of `2^20` states, which is why optionality is one of the two multipliers that
push a real schema engine to a PDA. **Nesting is the other**, and it is the one a DFA cannot absorb
at any size.

### 5.5 Token healing

```
prefix     = text of the token the grammar forced
candidates = [i for i, t in enumerate(vocab) if t.startswith(prefix)]
best       = argmax(probs[i] for i in candidates)
```

The worked case: the grammar requires `:` next, and the model — trained on URLs tokenized as a unit —
wants `://`. Masking leaves `:` at `p = 0.02`. Healing rebuilds the candidate set as `{":", ":/",
"://"}`, finds `://` at `p = 0.62`, and advances three characters in one step. The probability gain
over the mask route is `+0.60`, and the surface form is `'://'` on both routes — the assertion
`surface_preserved` is True because healing changed the *tokens*, not the *string*: "we're not
actually changing our output string. We're just changing the tokens of the output to get there" `[T]`.

Two details the production code must keep:

- **The candidate set must be rebuilt from the prefix, not from the mask.** The mask's legal set is
  `{":"}` — every multi-character token was excluded before healing ran. Healing is a *second*
  candidate construction over the full vocabulary, and using the masked set as its input makes it a
  no-op.
- **`changed` is reported.** When the argmax of the healed set is the token the mask would have
  chosen anyway, healing spent a recompute and bought nothing. That counter is the evidence for
  whether the trigger heuristics are earning their cost.

The four trigger heuristics are the corpus's list, evaluated rather than assumed `[T]`: a curated
offender list, "when the last token is very short", "when the last token doesn't start or end with
whitespace or punctuation", and "when the last token predicted was an exact prefix of another likely
token". In the run, `:` fires all four; `abc` fires `no_boundary_whitespace` only; `/` fires
`offender_list, very_short`. There is no single accepted rule, so the list is versioned and its firing
rates measured rather than trusted.

### 5.6 Cost of the healing caveat

`healing_cost(rate, tokens_per_record=400)`:

| Trigger rate | Triggered steps | Extra passes | Overhead |
|---|---|---|---|
| 0.00 | 0.0 | 0.0 | 0% |
| 0.02 | 8.0 | 8.0 | 2% |
| 0.10 | 40.0 | 40.0 | 10% |
| 1.00 | 400.0 | 400.0 | 100% |

The corpus's caveat — "it's just expensive. you have to go back and recompute" `[T]` — is a statement
about the last row. The design's claim is that the second row is the operating point and the trigger
heuristics are what keep it there; every point of trigger rate costs a point of overhead.

### 5.7 FUDGE: reward-augmented posterior with truncation

```
ids       = top_k_ids(probs, k)
posterior = softmax(log(probs[i]) + log(discriminator[i]))   over ids only
winner    = argmax(posterior)
```

The mechanism is `P(constraint | prefix) × P(token)`, and the exact form does not matter because
"we're going to softmax everything anyway. So, we don't care about sort of exact value" `[T]`. The
measured behaviour reproduces the structure of the corpus's worked example: over
`{do, you, want, prefer, thus, please, kindly}` with a formality discriminator, the unconstrained
winner is `do` and the constrained winner is `prefer` — the winner moves because the two were close
in language-model probability and the constraint separated them, which is the corpus's "want and
prefer were relatively even probability but prefer is more formal so it gets updated" `[T]`. The
extreme candidate `thus` has the highest formality score and still loses on low LM probability `[T]`.

**The blind spot is measured, not assumed.** With `k = 3` the truncation removes `prefer` from the
candidate set entirely, so the top-3 run picks `want` and cannot reach `prefer` at all; the discarded
constrained mass is printed, making the gap a number rather than an argument. The corpus's production
figure is `k = 200` against 100k+ `[T]`; our `k = 3` against 7 is the same phenomenon at a table-sized
scale.

### 5.8 Contrastive decoding

`adjusted = logits_expert - alpha * logits_amateur`, then softmax. Where the amateur's mass sits on a
memorised pair, the adjusted top token is not the expert's top token: the expert's `Honolulu` at
`p = 0.30` loses to `1961` at `p = 0.10`, because the amateur also likes `Honolulu` and only the
difference survives. The corpus's requirements — "identical tokenizers" and logit access on both
models — are what make the subtraction well-defined; the cost is a flat **2x** `[T]`, so it is gated
to surfaces where the amateur is already resident.

## 6. Concurrency & Locking

**Grammar state is per sequence, and that is the only hard rule.** The engine's logit processor is
called once per step for the whole batch; iterating a batch and applying sequence `i`'s mask to
sequence `j`'s logits is the failure mode that produces valid-looking records with another
customer's fields. The state map is keyed by sequence id and read through a per-sequence handle, not
by a batch index.

**Compiled grammars are immutable and shared.** The artifact is content-addressed and never mutated,
so concurrent reads need no lock. The only mutable structure is the cache's index, and a miss
compiles and inserts — a duplicate compile from a race is wasted work, not corruption.

**The `-inf` sentinel is a per-step buffer, not shared state.** `masked_logits` returns a new list;
the engine's own mask buffer is reused per step. Sharing a mutated logits tensor across sequences
would let one sequence's grammar leak into another's distribution within a batch.

**Healing is a per-sequence recompute.** Rolling back one position under paged attention means
releasing the KV entry for that position and re-running one forward pass for that sequence only. In a
batched step this is a *ragged* recompute and must not stall the batch — the production shape puts it
on a deferred path that rejoins at the next step.

**The validator is asynchronous to decoding.** It runs after the record is emitted and never blocks a
batch; its result routes the record rather than steering the ongoing decode. Steering is the repair
loop's job, and the repair loop starts a new request.

## 7. Error Handling

| Error | Detected by | Action |
|---|---|---|
| Dead-end state at compile | `audit_grammar` | Build failure; artifact never published |
| Stranded state at compile | `audit_grammar` reverse closure | Build failure |
| Zero survivors at runtime | Per-step assertion | `GRAMMAR_FAULT`; dead-letter; page |
| Grammar not found for a request | Lookup | Fall back to JSON-mode prompt + validator; count |
| Depth cap exceeded | Push-time check | Fall back to a non-recursive grammar, or reject to DLQ |
| Validator rejects, budget left | Validator | Repair loop with the error appended |
| Validator rejects, exhausted | Validator | Dead-letter |
| Healing produces the same token | `changed == False` | Counted; feeds the trigger-rate review |
| Healing would break the surface | `surface_preserved == False` | Abort the heal; use the mask route; page |

The distinction that matters: **`GRAMMAR_FAULT` at runtime is a compiler bug, and it must page.**
Falling back silently would mean records ship without the guarantee the audit was told about. The
fallback ladder exists for load, not for grammar defects.

## 8. Resource Accounting

Per step, per sequence: one index over the legal set (tens of entries), one masked softmax over the
vocabulary, one transition lookup. Against a decode step that reads model weights, this is noise
`[D]`.

Per record: `tokens × (mask + trigger_rate × heal)`. At 400 tokens and a 2% trigger rate that is
`400 + 8 = 408` forward passes against 400 — **2%**.

Per grammar: one artifact of `n_states × out_degree` transitions — 10 states for the flat two-field
schema, 449 for the 5-optional-field one. At `[D]` 4 bytes per entry the whole 240-schema estate is
comfortably resident.

Per surface: the semantic scorer is `constrained_steps × k` batched discriminator evaluations. At
`[D]` 40 constrained steps and `k = 200` that is one batched pass over 200 short sequences per step,
roughly **10%** on that surface.

The ordering this produces: **masking is free, healing is cheap, FUDGE is affordable where the
constraint is narrow, and a flat 2x is affordable only where the amateur is already loaded.**

## 9. Configuration Surface

The versioned bundle is `production/grammar-policy.yaml`. Its load-bearing fields, with defaults:

| Field | Default | Why it is a knob |
|---|---|---|
| `compiler.version` | pinned | Part of the grammar cache key; a fix must invalidate cached artifacts |
| `audit.strict` | `true` | `false` demotes a dead end to a warning — **never used in production** |
| `mask.on_empty_legal_set` | `dead_letter` | The alternative, `continue_unmasked`, breaks the guarantee |
| `healing.enabled` | `true` | Degradation rung 2 |
| `healing.max_rollback` | `1` | 2 is the corpus's "very rarely" case |
| `healing.offenders` | `[":", " ", "/"]` | Curated; tokenizer-version-dependent |
| `semantic.enabled` | `true` | Degradation rung 1 |
| `semantic.k` | `200` | The corpus's production truncation `[T]` |
| `semantic.min_mass` | `0.9` | If the top-k does not hold this much mass, widen or skip |
| `validate.mode` | `always` | The only setting that satisfies the audit |
| `repair.max_attempts` | `2` | Higher hides schema bugs |

`semantic.min_mass` is the guard against the §5.7 blind spot: below it the truncation has cut into
the body of the distribution. It does not fix the blind spot; it makes it detectable.

## 10. Observability Hooks

| Signal | Type | Alerts on |
|---|---|---|
| `mask_legal_set_size` | histogram, per step | A step far below the grammar's expectation — a schema change |
| `mask_mass_removed` | histogram, per step | A rising median — the model and the schema are diverging |
| `grammar_audit_failures_total` | counter, per compile | Any value > 0 |
| `grammar_fault_total` | counter, per request | Any value > 0 — pages |
| `healing_trigger_total{reason}` | counter | A reason's share moving; feeds the offender list |
| `healing_changed_total` | counter | The ratio to `healing_trigger_total` — low means the triggers are wrong |
| `semantic_truncated_mass` | histogram, per step | Below `min_mass` |
| `validator_reject_total{reason}` | counter | Any rise — the mask and the validator have diverged |
| `repair_attempts` | histogram | Bimodal at the cap — a schema/template mismatch |
| `dlq_depth` | gauge | Sustained growth |

The pairing that matters: **`validator_reject_total > 0` while `grammar_fault_total == 0` is not a
contradiction, it is the design working.** The record was valid against the grammar the request
carried and invalid against the schema now in force — a version skew, a tokenizer change, or a
healing edge case: exactly what the validator exists to catch and the mask cannot.

The second pairing: `healing_trigger_total` high with `healing_changed_total` low means the triggers
fire where healing is a no-op — pure overhead, and the offender list is the fix.

## 11. Test Strategy

**Grammar unit tests (run in CI, on every schema change).**
1. Every compiled grammar passes `audit_grammar`: no dead ends, no stranded states.
2. The fixture record round-trips: `dfa.accepts(record)` is True and `validate_schema(record)`
   returns `ok`. Two implementations, one record.
3. A negative fixture — a record with a wrong value — is rejected by both.
4. The compiler and the validator are exercised independently in CI, with no shared fixture loader.

**Mask tests.**
5. `softmax` on an all-`-inf` row returns zeros and never `NaN` — asserted directly, because this is
   the difference between a counted fault and a poisoned batch.
6. For a valid record, `survivors` at every step is non-empty and the legal set size matches the
   grammar's expectation.
7. `renormalisation_loss` sums to 1 across kept and removed mass.

**Healing tests.**
8. `surface_preserved` is True for every heal over a fuzz corpus of prefixes.
9. `changed == False` exactly when the healed argmax equals the masked argmax.
10. The candidate set for a prefix contains every token starting with it, independent of the mask —
    the regression test for the "rebuild from the prefix" bug in §5.5.

**Semantic-scorer tests.**
11. The posterior sums to 1 over the truncated set; the untruncated winner is reachable at `k = n`.
12. `truncated_reward_mass` decreases monotonically in `k` — the truncation is doing what it claims.

**End-to-end.**
13. A record that fails validation enters the repair loop, and a repair budget of 0 sends it to the
    DLQ — never to the ERP.
14. The mask is never disabled by any configuration, asserted over the config schema itself.

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt` — the `-inf`-before-softmax mechanism verbatim; the schema-to-state-machine construction; the ~10-of-100,000 sparsity; the syntactic/semantic and token-wise/end-verifiable taxonomy; the hand-drawn FSA defect list; the regular/context-free/Turing hierarchy; the "push down automa not FSAs" claim; token healing's rollback procedure, surface invariant and four triggers; FUDGE's `P(constraint|prefix) × P(token)` rewrite, top-200 truncation, and the "want and prefer" worked example; contrastive decoding's identical-tokenizer requirement and flat 2x.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` — softmax and renormalisation, and the tail that makes the legal set sparse.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — paged KV and the cost of a rollback recompute inside a batch.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
