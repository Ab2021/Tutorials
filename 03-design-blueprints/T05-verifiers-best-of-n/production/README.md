# T05 — Production Artifacts (reference-grade)

> **Not executed by this repo.** Deployment shape for the selection service in
> [../HLD.md](../HLD.md) and [../LLD.md](../LLD.md). Costs are in **normalised units, never
> currency** — the corpus asserts no vendor price. `[T]` transcript · `[R]` repo · `[D]` derived.

```
production/
  README.md              this file
  verifier-registry.yaml the tiers, their versions, and the agreement gate
  selection.yaml         n, the diversity floor, the cascade, the tie-break
  judge-policy.yaml      the bias mitigations, as configuration
  agreement-eval.yaml    the gold set and the gate that refuses an unmeasured verifier
  k8s-topology.yaml      why the scorer replicas are separate from the engine
```

---

## 1. `verifier-registry.yaml` — the tiers and the trust gate

```yaml
# The registry is the ONLY source of verifier identity at serving time. A version skew
# across replicas produces inconsistent selection, which is worse than an outage because
# it is invisible.
verifiers:
  - name: policy-checks
    tier: 0
    kind: programmatic
    version: 3
    # Exact and ~free. ALWAYS run first: [T] "a programmatic check beats a learned one".
    cost_per_call: 0.0
    rules:
      - cite_real_policy        # does the draft cite a policy that exists
      - no_entitlement_promise  # does it promise what the account is not entitled to
      - no_prohibited_phrase
      - customer_name_matches_crm

  - name: outcome-rm-v2
    tier: 1
    kind: orm
    model_id: "orm-v2"
    version: 2
    cost_per_call: 0.02        # [D] normalised; one forward, no decode
    agreement: 0.81            # MEASURED vs the gold set. See agreement-eval.yaml
    min_agreement: 0.75

  - name: judge-cross-family
    tier: 3
    kind: judge
    model_id: "judge-b"        # a DIFFERENT FAMILY from the generator — see judge-policy.yaml
    prompt_hash: "sha256:..."
    version: 4
    cost_per_call: 1.00        # [D] normalised; one GENERATION
    agreement: 0.78
    min_agreement: 0.75

# The loader refuses to start if any served verifier has agreement: null.
profiles:
  production:
    require_agreement: true    # a verifier you have not measured is not a verifier
  development:
    require_agreement: false   # permitted only here
```

**`require_agreement: true` in production is the single most important line in the file.** Every
other knob degrades quality; an unmeasured verifier silently *misrepresents* it — the system reports a
selection as verified when nothing verified it.

---

## 2. `selection.yaml` — n, the gate, the cascade

```yaml
selection:
  n: 32                # [T] chosen because it FITS ONE BATCH; the 33rd needs a second pass
  n_cap: 32            # ON the boundary, never above it
  temperature: 0.8     # [D] low temperatures collapse the pool [T] at 0.2, 100 draws -> ~20 unique
  diversity_floor: 0.5 # [D] unique fraction; the gate fires BEFORE any verifier call
  tie_break: shorter   # [D] non-arbitrary: judges prefer long, so ties skew long

cascade:
  # Tier 0 on all -> tier 1 on survivors -> tier 3 on the top k.
  tier0: all
  tier1: survivors
  tier3: top_k
  top_k: 4             # [D] a RECALL parameter, not a cost parameter.
                       # Set it by measuring: how often is the human-preferred candidate in
                       # the ORM's top-k? Raising k costs judge generations linearly.

degradation_cascade:
  # The order quality is given up under load. Tier 0 is NEVER in this list.
  order: [tier3, tier1, n]
  never: [tier0_exact, audit_record, agreement_check]
```

**What degradation must not do.** It may reduce `n`, and it may drop the judge and fall back to the
ORM. It may **not** return a selected candidate while reporting a verification that did not happen.
Losing the judge is a quality regression; claiming a judge that did not run is a correctness lie, and
the corpus's whole framing of the verifier is that it is the thing being trusted.

---

## 3. `judge-policy.yaml` — bias mitigations as configuration

```yaml
# The corpus names three judge biases (Jung et al. 2023): position, length, self-preference. [T]
# For a SELECTION system the dangerous one is length, and it is dangerous in a direction:
# the judge prefers long, so best-of-N SELECTS FOR LENGTH even when length is uncorrelated
# with quality. The symptom is drafts that inflate over time and read as thoroughness.
judge:
  family: "judge-b"
  generator_family: "gen-a"
  require_cross_family: true    # [D] same-family judging prefers its own family's outputs

  mitigations:
    randomise_position: true    # [D] kills position bias; record the permutation used
    length_normalise: true      # [D] kills verbosity bias
    pin_max_length: null        # [D] set a hard cap if the task has one

  # Alerting on the silent failure. Quality is flat while length rises; the ONLY visible
  # symptom is the length distribution of CHOSEN outputs.
  alerts:
    - name: chosen-length-inflation
      metric: mean_length_of_selected
      condition: rising over 7d with flat quality
      action: page — the judge is selecting on length
```

**The alert is the design's only detection for verbosity bias**, because every other dashboard shows
a healthy system: quality flat, latency flat, cost flat, drafts longer.

---

## 4. `agreement-eval.yaml` — the gate that refuses an unmeasured verifier

```yaml
gold_set:
  path: "s3://.../gold/review-decisions.jsonl"   # human labels, not model labels
  size: 400                                       # [D]
  sampling: "stratified by task_class and difficulty"
  refresh: monthly                                # labelled preferences drift

agreement:
  metric: "|scorer agrees with human| / |gold set|"   # [D]
  min: 0.75                                            # [D] per-verifier floor
  gate:
    # Re-measure on EVERY change to: model_id, prompt_hash, version.
    trigger: [model_id, prompt_hash, version]
    on_below_min: block_deploy
  live_check:
    # The gold set is fixed; production traffic is not. A scorer can hold agreement on
    # the gold set while drifting on live traffic — which is what reward hacking looks
    # like from the outside. Sample live traffic for periodic human labelling.
    enabled: true
    sample_rate: 0.01
```

**The two-stage check is deliberate.** A held-out gold set catches a bad scorer at deploy time. It
does **not** catch a scorer that a best-of-N loop has slowly learned to game — that is precisely what
the KL bound describes, and the defence is continued labelling of live traffic rather than a bigger
frozen set.

---

## 5. `k8s-topology.yaml` — why the scorers are separate

```yaml
# A reward model is a DIFFERENT model from the generator: different memory footprint,
# different batching profile (one forward per candidate, no KV growth, no decode loop).
# Co-locating them on the generator's replicas forces ONE scheduler to satisfy two
# incompatible shapes. Same argument T12 makes for prefill/decode disaggregation, one level up.
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: reward-model}
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: orm
          resources: {limits: {nvidia.com/gpu: 1}}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: judge}
spec:
  replicas: 0                      # scale-to-zero when the cascade drops tier 3 under load
  template:
    spec:
      containers:
        - name: judge
          # The judge SERIALISES per decision: position-randomise once, before fan-out, and
          # record the permutation. Two concurrent judge calls sharing a comparison context
          # have an undefined ordering, and the ordering is what position bias acts on.
          env: [{name: JUDGE_POLICY, value: /etc/judge/judge-policy.yaml}]
```

**`replicas: 0` on the judge is the degradation cascade made physical.** Under load the service drops
to the ORM-only path; the judge scales to zero and its cost disappears entirely. That is the
difference between "we degraded the verifier" and "we turned off a tier we had explicitly bounded".

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt` — the tier hierarchy, exact-first, `n = 32`, post-hoc filters
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` — critic reranking 20→32% at ~16×
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt` — position, length and self-preference bias; agreement measurement

**All configuration is `[D]`** except where a comment carries `[T]`. The corpus supplies the
mechanisms, the biases and the batch-fit argument; it supplies no configuration file.
