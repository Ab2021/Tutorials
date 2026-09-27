# Design Blueprints

System designs for operating inference infrastructure. One folder per topic.

This is the **design-first** artifact family: `HLD.md` and `LLD.md` carry the weight. Code is thin
and exists only to prove that the design's central mechanism actually works — not to be a library.

---

## What is in a blueprint

```
Tnn-<slug>/
  HLD.md               High-Level Design — context, containers, components, data flow,
                       topology, scaling, failure domains, capacity model, decisions
  LLD.md               Low-Level Design — modules, data structures, interfaces, state
                       machines, algorithms, concurrency, errors, config surface, tests
  docs/SEQUENCES.md    Full sequence diagrams — cold start, steady state, cache hit/miss,
                       failure, recovery, scale-out/in
  run.py               Entry point. Runs offline. Prints real numbers.
  sim/                 The minimum code needed to make run.py mean something
  production/          Real-world configs, reference-grade (see below)
```

Read the HLD for *what and why*, the LLD for *how*, `docs/SEQUENCES.md` for *in what order*, and
run `run.py` to see the mechanism produce numbers.

```bash
cd T08-batching-scheduling
python run.py
```

---

## Simulated vs reference-grade

Nothing here needs a GPU, and nothing here was executed against a GPU.

**`run.py` + `sim/` — executed and verified.** A small, self-contained model of the design's key
mechanism: a block allocator, a batching scheduler, a spec-decode acceptance loop, a route scorer.
Stdlib-only where possible. Its numbers are the *simulation's* output under stated assumptions —
they are **not** measurements of any real engine. Each one prints its assumptions so you can see
what it is and is not telling you.

**`production/` — reference-grade, not executed.** Real vLLM launch flags, llm-d Helm values,
Kubernetes manifests, NIXL transfer config, OTel collector pipelines — the shape you would actually
deploy. Every file opens with a comment saying it was **not** run in this environment. Treat it as
a well-informed starting point to adapt, not as validated configuration.

**Nothing in this folder is a benchmark.** Benchmarks live in the case studies and are attributed
to the speaker who reported them. See [`../README.md`](../README.md#on-numbers).

---

## The blueprints

**Layer A — Decoding Algorithms**
- [`T01-sampling-decoding`](T01-sampling-decoding/) — sampler design, parameter surface, reproducibility
- [`T02-search-decoding`](T02-search-decoding/) — beam/A* search over sequences, hypothesis clustering
- [`T03-constrained-generation`](T03-constrained-generation/) — grammar engine, FSM compilation, logit masking
- [`T04-test-time-compute`](T04-test-time-compute/) — reasoning budget controller, self-consistency fan-out
- [`T05-verifiers-best-of-n`](T05-verifiers-best-of-n/) — verifier pool, Best-of-N scheduler, reward serving

**Layer B — Engine Internals**
- [`T06-inference-fundamentals`](T06-inference-fundamentals/) — performance model, latency budget, metrics pipeline
- [`T07-kv-cache`](T07-kv-cache/) — block allocator, prefix cache, offload tier manager, retention API
- [`T08-batching-scheduling`](T08-batching-scheduling/) — continuous-batching scheduler, chunked prefill, fairness
- [`T09-speculative-decoding`](T09-speculative-decoding/) — draft/target loop, acceptance accounting
- [`T10-quantization`](T10-quantization/) — calibration pipeline, accuracy gate, KV-cache quant

**Layer C — Distributed & Serving Infrastructure**
- [`T11-parallelism-moe`](T11-parallelism-moe/) — parallel-plan solver, MoE all-to-all, topology mapping
- [`T12-disaggregation-kv-transfer`](T12-disaggregation-kv-transfer/) — prefill/decode pools, KV transfer plane
- [`T13-serving-engines`](T13-serving-engines/) — engine runtime, llm-d control plane, agent substrate
- [`T14-routing-gateways`](T14-routing-gateways/) — semantic router, prefix-aware performance router, gateway
- [`T15-autoscaling-slo`](T15-autoscaling-slo/) — capacity model, autoscaler, admission control, SLO budget

**Layer D — Operations, Agents & Governance**
- [`T16-agentic-inference`](T16-agentic-inference/) — agent loop runtime, tool/memory plane, durable sessions
- [`T17-observability-evals`](T17-observability-evals/) — trace pipeline, eval harness, judge service
- [`T18-guardrails-security`](T18-guardrails-security/) — rail pipeline, sandbox broker, identity & delegation
- [`T19-finops-sovereignty`](T19-finops-sovereignty/) — cost attribution, unit economics, sovereignty controls
