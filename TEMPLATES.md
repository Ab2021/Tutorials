# Authoring Templates

Mandatory structure for every artifact. Sections marked **required** must be present even if the
content is "not applicable — with reason."

**Conventions used throughout**

- Provenance markers **`[T]`** / **`[R]`** / **`[D]`** on every non-obvious claim (see
  [`README.md`](README.md#provenance-legend)).
- Header block on every file, immediately under the H1.
- Cross-references use the topic ID: `[T07](../01-case-studies/T07-kv-cache.md)`.
- Mermaid for all diagrams. Tables for all comparisons.
- `## Sources` block last, naming **real** files under `refs/`.
- Never invent a benchmark. Cite the speaker. Label your own arithmetic as yours.

---

## 1. Case study — `01-case-studies/Tnn-slug.md`

````markdown
# Case Study: <Title>

> **Topic:** `Tnn` · **Transcript coverage:** primary|partial|none · **Difficulty:** L3|L4|L5
> **One line:** <the decision this case study is really about>

## Table of Contents
<generated list>

---

## 1. The Scenario
Concrete, named situation. Who, what scale, what constraint. Not a generic "imagine a company".
Include the political/organisational constraint too — they are usually the real blocker.

## 2. Requirements
### Functional
| Requirement | Priority | Notes |
### Non-functional  — **numbers, always**
| Requirement | Target | Rationale |
### Constraints and non-goals
What we are explicitly *not* solving, and why. <provenance marker>

## 3. Architecture
```mermaid
flowchart ...
```
Prose walkthrough: what each box does, what crosses each arrow, what the request path is.

## 4. Component Deep Dive
For each major component: the mechanism, the alternatives, the failure behaviour.
This is where the transcript material lands — quote the talk where it is sharp.

## 5. Decision Table  ← the heart of the case study
For **every** significant design choice:

### 5.N <Decision name>
| Option | Pros | Cons | Exceptions — when it breaks | When to use |
|---|---|---|---|---|
| A | | | | |
| B | | | | |

**Chosen:** A, because <reason tied to the requirements in §2>.
**Revisit if:** <the condition that would change this decision>.

## 6. Edge Cases & Exceptions
The situations that break the clean model: adversarial input, degenerate shapes, cold start,
multi-tenant interference, version skew, partial failure. Each with the symptom and the handling.

## 7. Failure Modes & Mitigations
| Failure | Symptom | Detection | Blast radius | Mitigation | Recovery |
|---|---|---|---|---|---|

## 8. Capacity & Cost Model
**Worked arithmetic with assumptions shown.** Every step re-derivable.
- Assumptions table (traffic shape, model size, hardware, prices)
- Step-by-step calculation
- Sensitivity: what the answer is at 1x / 10x / 0.1x
- Break-even analysis where relevant

## 9. Benchmarks & Measured Numbers
Only real figures, each attributed.
| Metric | Value | Source | Conditions |
Mark vendor claims as vendor claims.

## 10. Operational Runbook
**Deploy** · **Tune** (which knobs, in what order, and what each moves) · **Monitor** (the
dashboard, the alert thresholds) · **Incident** (top 5 incidents, symptom→diagnosis→action).

## 11. What Changes at 10x
Which decisions survive, which invert, and the first thing that breaks.

## 12. Interview Walkthrough
How to present this in 35 minutes. The whiteboard order, the two numbers to say out loud, the
tradeoff to volunteer before you are asked. Then 5–8 follow-up questions with answers.

## Sources
````

---

## 2. Interview bank — `02-interview-questions/Tnn-slug.md`

````markdown
# Interview Bank: <Topic>

> **Topic:** `Tnn` · **Transcript coverage:** … · **Questions:** 25–30 · **Levels:** L3–L5
> Companion case study: [Tnn](../01-case-studies/Tnn-slug.md)

## How this bank is graded
Level definitions, and what "signal" means here.

## Table of Contents
Grouped by sub-theme, with question numbers.

---

## <Sub-theme>

### Tnn-Q1 — <question>
**Level:** L4 · **Expected depth:** <30s / 2min / whiteboard>

**Model answer.**
<The answer a strong candidate gives. Multi-paragraph where the topic deserves it. Numbers.>

**Signal.** <What distinguishes a strong answer from a memorised one.>

**Follow-ups.**
- <question> — <what it tests>

**Red flags.** <Answers that indicate shallow understanding.>

---

## Whiteboard Exercises
### Exercise 1 — <title>
Prompt, what a good whiteboard looks like, the traps.

## Sources
````

Target 25–30 questions per topic. Mix: ~40% L3 (fundamentals), ~45% L4 (design/tradeoff),
~15% L5 (open problems, "what would you do if…").

---

## 3. Cheat sheet — `00-cheat-sheets/Tnn-slug.md`

**One dense page. No padding. Every line load-bearing.**

````markdown
# Cheat Sheet: <Topic>

> `Tnn` · [Case study](../01-case-studies/Tnn-slug.md) · [Blueprint](../03-design-blueprints/Tnn-slug/HLD.md)

## Numbers to know
| Quantity | Value | Source |
Only figures that actually matter. Attributed.

## Decision matrix
| Situation | Do this | Not this | Why |
The one-table summary of the whole topic.

## Formulas
Each with variable definitions and a one-line worked example.

## Configuration
```<lang>
# the flags that matter, with the value you would actually set and a comment
```

## Failure signatures
| Symptom | Likely cause | First check |
Debugging table. This is the most-used section.

## Gotchas
Bulleted, each one sentence, each something that bites in production.

## Commands / API
The handful you actually type.

## Sources
````

---

## 4. Design blueprint — `03-design-blueprints/Tnn-slug/`

**This is the priority artifact family.** Design-first: HLD and LLD carry the weight, code is thin
and exists only to prove the mechanism. "Not much code" is the instruction.

### `HLD.md` — High-Level Design

````markdown
# HLD: <Topic>

> `Tnn` · [LLD](LLD.md) · [Case study](../../01-case-studies/Tnn-slug.md) · [Cheat sheet](../../00-cheat-sheets/Tnn-slug.md)

## 1. Problem & Scope
What system this designs. In scope / out of scope.

## 2. Requirements
Functional, non-functional (numbers), constraints, explicit non-goals.

## 3. System Context (C4 L1)
```mermaid
flowchart LR
```
Actors, external systems, what crosses the boundary.

## 4. Container View (C4 L2)
```mermaid
flowchart TB
```
Each container: responsibility, technology, scaling unit, state it owns.

## 5. Component View (C4 L3)
```mermaid
flowchart TB
```
Only for the containers that carry real design risk.

## 6. Data Flow
The request path end to end, numbered, with what is transformed at each hop.
Sequence diagram for the two or three paths that matter.

## 7. Deployment Topology
Node/GPU layout, network fabric, what is co-located and why.

## 8. Scaling Strategy
Per dimension (requests, tokens, context length, tenants, models). Which are horizontal, which
vertical, which do not scale at all.

## 9. Failure Domains & Degradation
What fails independently. What the system does when each piece is gone. Graceful degradation ladder.

## 10. Capacity Model
Worked arithmetic: given the requirements in §2, how many GPUs / nodes / replicas. Assumptions shown.

## 11. Key Design Decisions
| Decision | Options | Chosen | Why | Revisit if |

## 12. Build vs Buy
What to build, what to adopt (vLLM / llm-d / Ray / KServe / managed API), and the break-even.

## Sources
````

### `LLD.md` — Low-Level Design

````markdown
# LLD: <Topic>

> `Tnn` · [HLD](HLD.md) · [Sequences](docs/SEQUENCES.md)

## 1. Module Map
```mermaid
flowchart LR
```
Every module: responsibility, what it owns, what it must never do.

## 2. Core Data Structures
Each with its fields, invariants, and lifecycle. The structures that make the design work —
block tables, request state, scheduling queues, routing tables, session records.
```python
# illustrative shape, not code to run
```

## 3. Interfaces & Contracts
### 3.N `<interface name>`
Signature, parameters, returns, error modes, idempotency, threading. One block per interface.
Preconditions / postconditions / invariants stated explicitly.

## 4. State Machines
```mermaid
stateDiagram-v2
```
Every entity that has states, with transitions, guards, and what triggers each.

## 5. Algorithms
The non-obvious ones, in pseudocode with complexity. Block allocation, eviction, scheduling
policy, scoring, acceptance.

## 6. Concurrency & Locking
What runs concurrently, the lock discipline, lock-free paths, ordering guarantees, memory model
concerns.

## 7. Error Handling
Error taxonomy, which errors are retryable, backoff, circuit breaking, what the caller sees,
what gets logged.

## 8. Resource Accounting
Every resource acquired, where it is tracked, how it is released, and what happens on abort.

## 9. Configuration Surface
| Knob | Type | Default | Range | Effect | Tuning order |
Every knob the design exposes, and what it moves.

## 10. Observability Hooks
Every metric, log line and span this design emits, with its name and what it tells you.

## 11. Test Strategy
Unit, integration, load, chaos. What each proves. The specific invariants worth asserting.

## Sources
````

### `run.py` + `sim/` — the thin runnable core
Runs offline on Windows with **stdlib only** where possible. Proves the design's central mechanism
and prints real numbers. Keep it small — a few hundred lines total is right. No GPU.

### `production/` — reference-grade configs
Real vLLM launch flags, llm-d Helm values, K8s manifests, NIXL transfer config, OTel collector
pipeline — as used in production. Every file opens with a comment stating it is **reference-grade
and has not been executed in this environment**.

### `docs/SEQUENCES.md`
The full sequence diagrams — cold start, steady state, cache hit, cache miss, failure, recovery,
scale-out, scale-in.
