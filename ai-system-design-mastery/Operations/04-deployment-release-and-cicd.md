# Deployment, Release and CI/CD for AI Systems

**TL;DR** — Classical CI/CD assumes a deterministic artefact: tests pass or fail, the binary is the binary, rollback restores the previous binary. AI systems break all three assumptions. **Your artefact is four things that version independently** — code, prompt, model, and index — tests are **statistical**, not binary, and a rollback must restore a *combination* of versions, not one. This playbook covers the release pipeline, the quality gates with real thresholds, shadow and canary deployment for non-deterministic systems, the rollback procedure that actually works (version pinning, not redeployment), and the change-management rules that keep prompt edits from quietly becoming production changes. The organising insight: **an AI release is a configuration change to a pinned set of four artefact versions, and the only reliable rollback is a pointer flip.**

> **60-second interview answer:** "The core difference is that in a normal service my deployable artefact is one thing — the build. In an LLM system it is four things that version independently: the code, the prompt, the model version and the retrieval index. So the first thing I do is make them all part of one release manifest, pinned by a single release ID. The second difference is that my tests are statistical, not binary, so my gates are metric thresholds rather than pass/fail — golden set accuracy at 0.90, faithfulness at 0.85, no regression greater than two points on any slice. The third difference is deployment: I can't just canary a model change the way I canary a stateless container, because the outputs differ in kind, not just in error rate. So I use shadow first — run the new version on mirrored traffic, compare outputs offline, and only then canary at 5% with a 30-minute bake. And rollback is a pointer flip to the previous release ID, not a rebuild, because rebuilding takes 20 minutes and a bad prompt needs to be out in 90 seconds."

---

## 1. Purpose

This playbook defines how a change reaches production in an AI system, and how it gets out again when it goes wrong. It exists because the standard software release process is **structurally insufficient** for LLM systems in four specific ways:

| Assumption classical CI/CD makes | How AI systems break it |
|---|---|
| The artefact is deterministic | Same input, different output across calls |
| Tests are pass/fail | Quality is a distribution, measured statistically |
| One version to roll back | Four versions, coupled, that must be rolled back together |
| Failures are loud | Quality degradation is silent — latency and error rate stay green while answers get worse |

The last one is the reason this playbook exists. **A prompt change that makes answers 8% less faithful produces zero errors, zero latency change, and no alert.** Nothing in a classical monitoring stack catches it.

**In scope:** prompt changes, model version changes, code changes to the orchestration layer, retrieval index changes, parameter changes, and the pipeline that gates them.

**Out of scope:** infrastructure provisioning (see [capacity planning](05-capacity-planning-and-autoscaling.md)), fine-tuning job execution (the *release* of a tuned adapter is in scope), and incident response for infrastructure outages (see [reliability patterns](07-reliability-patterns.md)).

---

## 2. Scope — what counts as a release

**Every one of these is a release and must go through the pipeline:**

| Change | Risk | Why | Typical blast radius |
|---|---|---|---|
| **System prompt text** | **High** | Changes behaviour globally, instantly, on every request | 100% of traffic |
| **Model version** | **High** | Different training, different failure modes | 100% of traffic |
| **Model provider / endpoint** | **High** | Different tokeniser, different latency profile | 100% of traffic |
| **Retrieval index rebuild** | Medium | Changes what the model sees | 100%, but bounded by retrieval quality |
| **Chunking / embedding model** | **High** | Full re-embed, all retrieval shifts | 100% |
| **Reranker model** | Medium | Reorders evidence | 100% of ranked results |
| **Tool schema** | Medium | Agent may call differently | Agent traffic |
| **Temperature / top_p** | Medium | Variance changes | 100% |
| **Guardrail policy** | **High** | Can block or allow wrongly | 100% |
| **Code (orchestration)** | Medium | Ordinary software risk | Varies |
| **Cache TTL / threshold** | Medium | Staleness vs cost | 100% |

**The rule that follows:** if it changes model behaviour, it is a release and it gets a version. **A prompt edited directly in a production config map is an uncontrolled release**, and it is the single most common source of "how did this change?" incidents.

### 2.1 The release manifest — the central artefact

```yaml
# release.yaml — one file, one release ID, four artefact versions
release_id: rel_2026_09_27_003
created_at: 2026-09-27T14:02:11Z
created_by: ci-pipeline
git_sha: 8f3c1a9

code:
  image: registry/acme/rag-service:8f3c1a9
  digest: sha256:4a7b...   # digest, never a mutable tag

prompt:
  bundle_id: prompts_v47
  hash: sha256:9c1e...
  system_prompt: v31
  templates: { answer: v12, rerank: v4, triage: v9 }

model:
  generation:
    provider: bedrock
    model_id: anthropic.claude-sonnet-4-6
    snapshot: 2026-08-14          # PINNED, never a floating alias
    params: { temperature: 0.0, max_tokens: 1024, top_p: 1.0 }
  rerank: { model_id: bge-reranker-v2-m3, revision: "a1b2c3" }
  embed:  { model_id: text-embed-3-large, dims: 256, revision: "2026-03" }

index:
  collection: kb_primary
  version: idx_2026_09_27
  doc_count: 10_412_887
  built_at: 2026-09-27T11:40:00Z
  embedding_model: text-embed-3-large@2026-03

guardrails:
  policy_version: gr_v18
  injection_classifier: prompt-guard-2@2026-07

config:
  retrieval: { k: 50, rerank_to: 5, hybrid: true, rrf_k: 60 }
  cache: { semantic_threshold: 0.93, ttl_s: 3600 }
  timeouts: { llm_ms: 12000, rerank_ms: 1500, total_ms: 30000 }
```

**Four rules that make this work:**

1. **Every field is immutable once released.** No `latest`, no floating aliases, no mutable tags. `anthropic.claude-sonnet-4-6` alone is not a version — it is a family. Pin the snapshot.
2. **The release ID is the only thing that changes at deploy time.** Deployment is a pointer write.
3. **Every request logs its `release_id`** in the trace, so any output can be tied to the exact combination that produced it.
4. **A release is reproducible.** Given a `release_id`, you can reconstruct the exact system.

**The single most valuable sentence in this document:** *"If you can't answer 'which prompt and which model snapshot produced this bad answer from last Tuesday', you do not have a deployment process, you have a hope."*

---

## 3. SLIs and SLOs

### 3.1 Release-process SLIs

| SLI | Definition | SLO | Why |
|---|---|---|---|
| **Deploy frequency** | Releases to production per day | ≥ 10 (target), ≥ 1 (floor) | Low frequency means big risky batches |
| **Lead time** | Commit → production | p50 < 45 min, p95 < 3 h | Long lead time kills iteration |
| **Change failure rate** | Releases causing rollback or hotfix | **< 10%** | The headline process metric |
| **MTTR (rollback)** | Detection → previous release serving 100% | **< 5 min**, p95 < 10 min | This is what limits blast radius |
| **Gate duration** | Full eval suite wall clock | < 25 min | Longer gates get bypassed |
| **Gate flake rate** | Gates failing then passing on rerun | < 3% | Above this, people stop trusting gates |
| **Unversioned changes** | Production changes with no `release_id` | **0** | Audit |

### 3.2 System SLOs that constrain the release

| SLI | SLO | Rollback trigger |
|---|---|---|
| Availability | 99.9% | Error rate > 5% for 5 min |
| p95 end-to-end latency | < 3 s | p95 > 4.5 s for 10 min |
| Golden-set accuracy | ≥ 0.90 | < 0.85 |
| Faithfulness (groundedness) | ≥ 0.85 | < 0.80 |
| Answer relevance | ≥ 4.0 / 5.0 | < 3.7 |
| Refusal correctness | ≥ 0.95 | < 0.90 |
| **Ungated safety action rate** | **0** | Any |
| PII leak rate | **0** | Any |
| Cost per request | ≤ $0.012 | > $0.020 for 30 min |

### 3.3 The error budget for releases

```
   Monthly error budget (99.9%)             = 43 min 12 s

   Budget consumed by a bad release:
     5% canary for 10 min, then detected     ≈  0.5 min  ✓ cheap
     Detected after full rollout, 8 min MTTR  ≈  8 min    ✗ expensive

   Rule: a canary with a 30-min bake at 5% costs 1/20th of the
   budget of a full-rollout failure. This is why the canary
   exists and why the bake time is not negotiable.

   Freeze rule: if > 60% of the monthly budget is consumed,
   only rollback-target releases are permitted — no new features.
```

---

## 4. Architecture

```
══════════════════════ CHANGE SOURCES ══════════════════════

  ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────┐  ┌────────┐
  │  Code  │  │ Prompt │  │ Model  │  │  Index  │  │ Config │
  │  PR    │  │  PR    │  │  PR    │  │ Pipeline│  │  PR    │
  └───┬────┘  └───┬────┘  └───┬────┘  └────┬────┘  └───┬────┘
      └───────────┴───────────┴────────────┴───────────┘
                            ▼
              ┌──────────────────────────────┐
              │  RELEASE MANIFEST BUILDER    │
              │  → release_id, all 4 pinned  │
              └──────────────┬───────────────┘
                             ▼
══════════════════════ CI GATES (blocking) ══════════════════

   ┌───────────────────────────────────────────────────────┐
   │ G1  STATIC   syntax · token budget · schema · lint    │  2 min
   ├───────────────────────────────────────────────────────┤
   │ G2  UNIT     determinism · parsing · tool contracts   │  3 min
   ├───────────────────────────────────────────────────────┤
   │ G3  GOLDEN   fixed set, exact + must_contain + judge  │  6 min
   ├───────────────────────────────────────────────────────┤
   │ G4  QUALITY  sampled LLM-judge, multi-dimension       │  8 min
   ├───────────────────────────────────────────────────────┤
   │ G5  SAFETY   injection · jailbreak · PII · unauth     │  5 min
   ├───────────────────────────────────────────────────────┤
   │ G6  COST     tokens/request · $/request · p95 latency │  1 min
   └───────────────────────────────────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  QUALITY GATE   │  ALL must pass
                    │  (fail = block) │
                    └────────┬────────┘
                             ▼
══════════════════════ CD STAGES ══════════════════════

    ┌──────────────┐
    │ 1. SHADOW    │  mirrored traffic, no user impact
    │ 24 h or 10k  │  compare vs production offline
    └──────┬───────┘
           │  divergence < 10%, no safety delta
           ▼
    ┌──────────────┐
    │ 2. CANARY    │  5% → bake 30 min → 25% → 50% → 100%
    │ per-step gate│  auto-rollback on threshold breach
    └──────┬───────┘
           │  all steps green
           ▼
    ┌──────────────┐
    │ 3. FULL      │  release_id pointer = 100%
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ 4. WATCH     │  24 h heightened monitoring
    │ (not done)   │  on-call owns it until the window closes
    └──────────────┘

══════════════════════ CONTROL PLANE ══════════════════════

   ┌───────────────────┐   ┌────────────────────┐
   │ RELEASE REGISTRY  │   │  FEATURE FLAGS     │
   │ release_id →      │   │  kill switch per   │
   │  manifest, status │   │  capability        │
   └─────────┬─────────┘   └──────────┬─────────┘
             │                        │
             └────────┬───────────────┘
                      ▼
          ┌────────────────────────┐
          │  TRAFFIC ROUTER        │  reads pointer, applies
          │  (the ONLY thing that  │  canary %, no restart
          │   changes at deploy)   │  needed
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐
          │  SERVING FLEET         │  reads release_id per
          │  logs release_id on    │  request from router
          │  EVERY request         │
          └────────────────────────┘
```

**The architectural claim to defend:** *"Deployment is a pointer write to the release registry. Nothing is rebuilt, nothing is restarted. That is what makes a 90-second rollback possible."*

---

## 5. Standard operating procedures

### SOP-1 — Standard release (prompt or code change)

```
PREREQUISITES
  □ Change is in a PR against the prompt/code repo — never edited in prod
  □ release.yaml updated with new versions
  □ Author has stated the intent and the expected effect
  □ Eval set has not changed (if it has, that is a separate PR — see SOP-5)

STEP 1  CI runs G1–G6 (25 min max)
        → any gate failure blocks. No override without an incident
          commander's written approval and a documented reason.

STEP 2  Review the eval diff
        □ Open the eval report. Read the FAILED cases, not the score.
        □ Ask: are the failures acceptable, or are they the same bug?
        □ Compare per-slice, not just aggregate:
             short queries · long queries · multi-hop · no-answer
             non-English · adversarial · per-tenant
        □ A 2-point aggregate gain that hides a 15-point loss on
          one slice is a REGRESSION, not an improvement.

STEP 3  Deploy to shadow
        □ Route 100% of production traffic as a mirror (no response
          returned to users)
        □ Run for 24 h or 10,000 requests, whichever is first
        □ Compare: divergence rate, quality delta, safety delta,
          cost delta, latency delta
        □ BLOCK if divergence > 10% without a quality improvement,
          or any safety signal moves adversely

STEP 4  Canary
        □ 5% traffic, bake 30 min
        □ Gate check at end of bake (see §8 alerts)
        □ 25% bake 30 min → 50% bake 30 min → 100%
        □ Any threshold breach → SOP-2 automatic rollback

STEP 5  Full rollout
        □ Flip the release pointer to 100%
        □ Confirm release_id appears in traces

STEP 6  Watch window — 24 h
        □ On-call owns the change for 24 h
        □ Compare daily quality metrics against the previous release
        □ Post the release summary in #ai-releases

TOTAL WALL CLOCK: ~27 h (dominated by shadow + watch)
TOTAL GATE COST: ~$40–120 in eval inference per release
```

**The shadow-duration trade-off:** *"24 hours is expensive in lead time. For low-risk changes — a config tweak, a template string — I reduce shadow to 2 hours or skip it and go straight to a 5% canary with a 60-minute bake. For system prompt or model changes I never skip shadow, because those are the changes whose failures are silent. The rule is: if the change can alter the *kind* of answer, shadow it. If it can only alter the *speed* of the answer, canary it."*

### SOP-2 — Rollback (the 90-second procedure)

```
TRIGGER: any rollback threshold breached, or a human decides.

TARGET: previous release serving 100% in under 5 minutes.

STEP 1  (t+0s)   Announce in #ai-incidents:
                 "Rolling back rel_X to rel_Y, reason: <metric>"

STEP 2  (t+10s)  Flip the release pointer:
                 $ ai-release rollback --to rel_2026_09_27_002
                 → router reads the new pointer within 5 s
                 → NO pod restart, NO rebuild, NO image pull

STEP 3  (t+30s)  Verify:
                 □ release_id in traces == rel_..._002
                 □ error rate returning to baseline
                 □ canary traffic is zero

STEP 4  (t+60s)  Confirm the breached metric is recovering.

STEP 5  (t+90s)  Freeze: block all releases except rollback targets
                 until root cause is understood.

STEP 6  (t+24h)  Post-incident review. The failed release is
                 quarantined; a new release requires a new
                 manifest, not a re-flip of the bad one.

POST-CONDITIONS
  □ Previous release fully serving
  □ Bad release quarantined in the registry
  □ Rollback recorded in the audit log with the trigger metric
  □ If the rollback was triggered by an alert, that alert's
    threshold is reviewed — was it too late? too sensitive?
```

**Why this works and a redeploy does not:**

| Approach | Time to restore | Why |
|---|---|---|
| Rebuild + redeploy previous image | 12–25 min | Image pull, startup, warm-up |
| Blue/green cluster swap | 3–8 min | Provisioning, health checks |
| **Release-pointer flip** | **< 60 s** | Nothing is rebuilt; only routing changes |

**The critical prerequisite nobody builds:** *"A pointer flip only works if the previous release's artefacts still exist and are still servable. So the rule is: keep the last 10 releases hot — image cached, prompt bundle present, model snapshot still available from the provider, and the previous index version still queryable. If the index was rebuilt in place, you cannot roll back retrieval at all. That is why indexes are versioned collections, not overwritten ones."*

**The index rollback trap to name explicitly:** *"The most common rollback failure in RAG systems is rebuilding the vector index in place. Then a rollback restores the old prompt but points at the new index, and you get a hybrid state that never existed in testing. Version your indexes as immutable collections with aliases, and roll back the alias."*

### SOP-3 — Model version upgrade

Model changes are the highest-risk release, because you cannot diff them.

```
STEP 1  PREPARE
        □ Request the pinned snapshot ID from the provider (not a family name)
        □ Confirm tokeniser compatibility — a changed tokeniser changes
          every token count, which changes every cost and every truncation
        □ Confirm context window and max output
        □ Confirm the deprecation date of the CURRENT model
          (you need to know your deadline)

STEP 2  OFFLINE EVAL — full suite, not sampled
        □ Golden set: accuracy, per slice
        □ Regression set: every case that EVER failed, to check it still fails
        □ Adversarial set: injection, jailbreak, PII
        □ Format-compliance: does it still produce valid tool-call JSON?
        □ Long-context: does behaviour at 100k tokens still hold?
        □ Cost model: recompute from real token counts — do NOT assume
          the new model tokenises the same way

STEP 3  SHADOW (72 h, not 24)
        □ Compare not just quality but the DISTRIBUTION of answers
        □ Look for a change in refusal rate — a model that refuses less
          may be better on the benchmark and worse on safety
        □ Look for a change in verbosity — output token growth is a cost bug

STEP 4  CANARY with an extended bake
        □ 5% for 2 h (not 30 min) — model differences surface slowly
        □ Then 25% / 50% / 100% at 1 h each

STEP 5  KEEP THE OLD MODEL WARM
        □ Do not decommission the previous snapshot until 7 days clean
        □ Rollback must remain possible for the full week
```

**The trap:** *"Model upgrades fail in the direction of 'still fine'. The new model is usually better on your benchmark, marginally worse on one slice you don't monitor, and 12% more verbose. You find out about the verbosity from the invoice a month later. So recompute cost from measured token counts during shadow, and alert on output-token-growth per release."*

### SOP-4 — Index rebuild and embedding-model change

```
STEP 1  Build the NEW index as a new immutable collection.
        NEVER rebuild in place.
        □ name: kb_primary__idx_2026_09_27
        □ record: doc count, chunk count, embedding model + revision,
          build duration, source snapshot timestamp

STEP 2  Dual-index evaluation
        □ Run the golden retrieval set against BOTH indexes
        □ Metrics: recall@50, nDCG@10, MRR@10, and — most importantly —
          the answer-level metrics through the full pipeline
        □ BLOCK if nDCG@10 drops > 3 points
        □ BLOCK if recall@50 drops > 2 points

STEP 3  Cost and size check
        □ New index memory footprint vs capacity headroom
        □ Query latency p95 vs the 100 ms retrieval SLO
        □ Re-embedding cost already incurred — log it

STEP 4  Alias flip (this is the deploy)
        □ ai-index alias set kb_primary -> kb_primary__idx_2026_09_27
        □ Old collection stays queryable for 14 days

STEP 5  Rollback = alias flip back
        □ Tested in staging before the production flip. Always.
```

**The dual-write window:** if documents are being ingested continuously, the new index must be kept current or it goes stale between build and flip. Either build from a frozen snapshot and accept a bounded staleness window, or dual-write to both indexes during the transition. **State the choice; do not leave it implicit.**

### SOP-5 — Changing the evaluation set

The eval set is production infrastructure. Changing it changes the meaning of every metric.

```
RULE: an eval-set change is its own PR, reviewed by someone other than
      the author, and it NEVER ships in the same release as the change
      it is meant to measure.

□ Adding cases: allowed, versioned (eval_v12 → eval_v13)
□ Removing cases: requires a written reason and a second reviewer
□ Changing expected outputs: requires a second reviewer — this is how
  a regression gets laundered into a passing test
□ Re-slicing: recompute all historical metrics on the new slices so
  trends remain comparable
□ NEVER tune the eval set to make a failing release pass. If you are
  tempted, that is the signal to roll back.
```

### SOP-6 — Emergency hotfix

```
WHEN: production is failing and the normal pipeline is too slow.

□ Requires an incident commander's approval — recorded in the incident
□ Minimal pipeline: G1 static + G3 golden + G5 safety only (~13 min)
□ Shadow: SKIPPED (documented as a deliberate exception)
□ Canary: 5% for 10 min, then 100%
□ Mandatory post-incident: the hotfix goes through the FULL pipeline
  retrospectively within 48 h, and if it fails, it is reverted
□ Audit: every emergency release is listed in the monthly ops report
```

**Cap the frequency:** *"More than two emergency releases in a month is not an incident problem, it is a process problem — it means the normal pipeline is too slow or too flaky, and I would fix the pipeline rather than keep waiving it."*

---

## 6. Failure modes and mitigations

| # | Failure | Symptom | Root cause | Mitigation |
|---|---|---|---|---|
| 1 | **Silent quality regression** | All infra metrics green; answers quietly worse | No quality signal in the deploy gate | G4 quality gate + shadow comparison; alert on answer-level metrics |
| 2 | **Prompt drift** | Prompt in prod ≠ prompt in repo | Prompt edited live in a config map | Prompts only in the registry, immutable, `release_id` in every trace |
| 3 | **Unpinned model alias** | Behaviour changes with no deploy | `model: latest` or a family alias | Pin snapshots; alert on snapshot-version mismatch |
| 4 | **Index rebuilt in place** | Rollback produces an untested hybrid state | Mutable index | Immutable versioned collections + alias flip |
| 5 | **Eval-set laundering** | Gate passes; real quality dropped | Expected outputs edited to match new behaviour | Eval changes are separate PRs with a second reviewer |
| 6 | **Flaky gate** | Gate fails, passes on rerun | Non-determinism + small sample | Fixed seed where possible; n ≥ 200 per gate; flake rate SLI |
| 7 | **Canary blind spot** | Canary green, full rollout bad | Canary traffic not representative | Canary on the same traffic mix; stratify by tenant and query class |
| 8 | **Rollback impossible** | Cannot restore; must roll forward | Previous artefacts decommissioned | Keep 10 releases hot; never decommission a model snapshot in < 7 days |
| 9 | **Cost regression** | Invoice spike | New model more verbose | Recompute cost from measured tokens; alert on $/request |
| 10 | **Tokeniser change** | Truncation, cost shift, format breakage | Model swap changed tokenisation | Verify token counts in shadow on real inputs |
| 11 | **Guardrail regression** | Injection succeeds after a policy edit | Guardrail policy shipped without the safety gate | G5 is blocking; policy version in the manifest |
| 12 | **Config/code mismatch** | Retrieval k=50 in code, 5 in config | Two sources of truth | Config in the manifest; code reads only the manifest |
| 13 | **Deploy during an incident** | New variable during troubleshooting | No freeze | Auto-freeze when an incident is open; only rollback targets allowed |
| 14 | **Cache poisoning across releases** | New prompt served from old cached answers | Cache key lacks `release_id` | **Cache key MUST include `release_id`** — flush on deploy |
| 15 | **Partial rollout stickiness** | Some users on old, some on new, indefinitely | Router hashing without a drain | Canary steps are time-boxed; a step that does not advance alerts |
| 16 | **Shadow traffic side effects** | Shadow wrote to the DB | Shadow path executed write tools | Shadow is read-only by construction; write tools stubbed |
| 17 | **Gate cost explosion** | Eval suite costs $2k per release | Full-judge on a huge set | Sampling + a small blocking golden set + a large sampled set |
| 18 | **Nobody reads the eval report** | Gates are rubber-stamped | Report is a number, not a diff | Report must show per-slice deltas and the top 10 changed cases |

### 6.1 The cache-key release bug — expanded

This one causes real production outages and is worth a dedicated paragraph.

```
   WRONG:  cache_key = hash(normalized_query)
           → after a prompt change, users get OLD answers for up to the TTL
           → "we deployed the fix but the bug is still happening"
           → on-call debugs the prompt for 40 minutes before finding it

   RIGHT:  cache_key = hash(release_id + normalized_query)
           → every release gets a cold cache
           → cold cache = a temporary cost and latency spike

   MITIGATION for the spike:
     □ Pre-warm the top 1,000 queries against the new release before
       flipping the pointer
     □ Accept a 15–30 min cost bump; budget for it
     □ Alert on cache hit rate dropping below 10% post-deploy,
       with the deploy as the expected cause
```

---

## 7. Capacity and cost of the pipeline itself

### 7.1 Gate cost per release

| Gate | Cases | Judge calls | Tokens | Cost |
|---|---|---|---|---|
| G1 Static | — | 0 | — | $0 |
| G2 Unit | 40 | 0 | ~20k | $0.02 |
| G3 Golden (blocking) | 200 | 0 (rules) + 60 judge | ~400k | $1.80 |
| G4 Quality (sampled) | 1,000 sampled 20% | 200 judge | ~1.2M | $6.40 |
| G5 Safety | 300 | 120 judge | ~380k | $2.10 |
| G6 Cost/latency | 50 | 0 | ~50k | $0.25 |
| Shadow (10k req) | 10,000 | 500 sampled judge | ~11M | **$48** |
| **Per release total** | | | | **~$59** |

**Fleet arithmetic:**

```
   Releases/day                          12
   Cost/release                          $59
   Daily pipeline cost                   $708
   Monthly                               $21,240

   vs. one bad release caught late:
     Full rollout, 8 min MTTR, degraded quality
     → ~4,000 affected requests, ~$0.02 credit each = $80
     → plus engineering time: 4 engineers × 2 h = 8 h
     → plus trust cost: unquantifiable but real

   One prevented bad release pays for ~3 weeks of pipeline cost.
   The gate is not expensive. It is the cheapest part of the system.
```

### 7.2 Pipeline throughput limits

| Resource | Limit | Mitigation |
|---|---|---|
| Concurrent eval jobs | 8 | Queue; releases wait |
| Judge model rate limit | 200 RPM | Batch, back off, spread |
| Shadow traffic capacity | 2x serving fleet | Size the fleet for shadow headroom, or shadow at 30% |
| CI runner minutes | 4,000/month | Cache the model client, mock in G1/G2 |
| Gate wall clock | 25 min budget | Parallelise G3/G4/G5; they are independent |

**The capacity insight:** *"Shadow doubles serving load. If your fleet has no headroom, shadow will cause the latency regression you were trying to detect. So either size for 2x, or shadow at 20–30% of traffic for longer. I usually shadow at 30% for 8 hours rather than 100% for 2.4 hours — same sample size, no capacity risk."*

---

## 8. Metrics and alerts

### 8.1 Release-process alerts

```yaml
release_alerts:
  # ── Deployment health ─────────────────────────────────────
  - name: canary_quality_regression
    condition: canary_quality_score < prod_quality_score - 0.05
    window: 30m
    severity: critical
    action: auto_rollback
    note: relative, not absolute — the canary is compared to prod

  - name: canary_error_rate
    condition: canary_error_rate > 0.05
    window: 5m
    severity: critical
    action: auto_rollback

  - name: canary_p95_latency
    condition: canary_p95_latency > 1.5 * prod_p95_latency
    window: 10m
    severity: critical
    action: auto_rollback

  - name: canary_cost_per_request
    condition: canary_cost_per_request > 1.5 * prod_cost_per_request
    window: 30m
    severity: warning
    action: pause_rollout

  - name: canary_safety_violation
    condition: canary_safety_violations > 0
    window: 1m
    severity: critical
    action: auto_rollback
    note: zero tolerance; no bake required

  - name: canary_stalled
    condition: canary_percentage_unchanged for 45m AND canary_percentage < 100
    severity: warning
    action: notify_release_owner

  # ── Post-deploy watch ────────────────────────────────────
  - name: post_deploy_quality_drift
    condition: quality_score_24h < pre_deploy_quality_score - 0.03
    window: 24h
    severity: warning
    action: page_release_owner

  - name: cache_hit_rate_drop
    condition: cache_hit_rate < 0.10
    severity: info
    action: none
    note: EXPECTED after a release — this is the cold-cache signal,
          not an incident. Documented so nobody pages on it.

  - name: output_token_growth
    condition: mean_output_tokens > 1.25 * pre_release_mean
    severity: warning
    note: the verbosity regression — catches cost drift early

  # ── Process integrity ────────────────────────────────────
  - name: unversioned_production_change
    condition: request_without_release_id > 0
    severity: critical
    action: page_platform

  - name: model_alias_drift
    condition: served_model_snapshot != manifest_model_snapshot
    severity: critical
    note: the provider changed the model under a floating alias

  - name: index_version_mismatch
    condition: queried_index_version != manifest_index_version
    severity: critical

  - name: gate_flake_rate
    condition: gate_flake_rate_7d > 0.05
    severity: warning
    note: gates people don't trust are gates people bypass

  - name: release_freeze_violation
    condition: release_started while incident_open
    severity: critical
```

### 8.2 Dashboards

**Dashboard 1 — Release health (on-call's first screen)**

| Panel | What it shows |
|---|---|
| Active release | `release_id`, age, canary %, bake time remaining |
| Quality delta | Canary vs prod, rolling 30 min, per metric |
| Error rate | Canary vs prod, overlaid |
| p95 latency | Canary vs prod, overlaid |
| Cost/request | Canary vs prod, overlaid |
| Safety counter | Cumulative violations, **must be 0** |
| Cache hit rate | With release markers annotated |

**Dashboard 2 — Release history (weekly review)**

| Panel | What it shows |
|---|---|
| Release timeline | Every release with outcome (success / blocked / rolled back) |
| Change failure rate | 30-day rolling, target < 10% |
| MTTR trend | p50 and p95 across rollbacks |
| Gate failure reasons | Pareto of why releases get blocked |
| Eval score trend | Per-metric over 90 days, per release |
| Unversioned changes | **Must be a flat zero line** |

---

## 9. Change management

### 9.1 Who can release what

| Change type | Author | Reviewer | Approver | Pipeline |
|---|---|---|---|---|
| Config tweak (TTL, k) | Any engineer | Peer | — | Full, shadow 2 h |
| Prompt template (non-system) | Any engineer | Peer | — | Full, shadow 24 h |
| **System prompt** | Prompt owner | **Second prompt owner** | — | Full, shadow 24 h |
| **Model version** | Platform | **Platform lead + eval owner** | **Eng lead** | Full, shadow 72 h |
| Index rebuild | Data owner | Platform | — | Dual-index eval |
| **Guardrail policy** | Safety owner | **Safety lead** | **Eng lead** | Full, safety gate blocking |
| Fine-tuned adapter | ML engineer | Eval owner | Eng lead | Full + extra eval slice |
| Emergency hotfix | On-call | **Incident commander** | IC | Minimal, retrospective |

### 9.2 The four-eyes rules

**These three changes always require two humans:**
1. **System prompt** — changes 100% of behaviour in one line of text.
2. **Guardrail policy** — a loosened guardrail is a security change.
3. **Evaluation set expected outputs** — this is how a regression gets hidden.

### 9.3 The release train

```
   10:00  Cut release branch. Manifest built. CI runs G1–G6.
   10:30  Eval report posted to #ai-releases. Reviewers read the DIFF.
   11:00  Shadow begins (or is skipped per the risk table).
   ...    Shadow runs 24 h (or 2 h for low-risk).
   11:00  +1d: Canary 5%, 30 min bake.
   11:35  Canary 25%, 30 min.
   12:10  Canary 50%, 30 min.
   12:45  Full rollout. Watch window opens.
   12:45  +1d: Watch window closes. Release closed.

   Anything not cleared by the cut time waits for the next train.
   Exception: hotfixes and rollbacks, which are never queued.
```

**Why a train rather than continuous deployment:** *"Continuous deploys of non-deterministic systems make attribution impossible. If three prompt changes and a model swap land in an hour and quality drops, you cannot tell which caused it. A release train batches changes so the release is the unit of attribution. It costs lead time, which is why I keep the train at twice a day and let hotfixes bypass it."*

---

## 10. Audit and compliance

### 10.1 What must be recorded, permanently

| Record | Contents | Retention | Why |
|---|---|---|---|
| **Release registry** | Every `release_id`, full manifest, author, approvers, timestamps | 7 years | "What was running on <date>?" |
| **Gate results** | Metrics per gate, pass/fail, the eval set version | 2 years | Prove the gate ran |
| **Eval reports** | Per-slice scores, failed cases | 2 years | Detect slow drift |
| **Rollback log** | Trigger, metric value, operator, MTTR | 7 years | Incident forensics |
| **Shadow comparisons** | Divergence, deltas, sample size | 1 year | Due diligence |
| **Approvals** | Who approved what, with timestamps | 7 years | Accountability |
| **Prompt bundle history** | Every prompt version, immutable | 7 years | Reproduce any output |
| **Request → release_id mapping** | Trace store | 90 days hot, 2 years cold | Tie an output to its cause |

### 10.2 The compliance questions you must be able to answer

| Question | Answer source |
|---|---|
| Which model version generated this response on 2026-03-14? | Trace `release_id` → registry → manifest |
| What prompt text was in production last month? | Prompt bundle history (immutable) |
| Who approved the guardrail change on that date? | Approval log |
| Was this change evaluated before release, and how? | Gate results + eval report |
| How long was the system degraded, and what was the blast radius? | Rollback log + request counts |
| Can you reproduce an output from 6 months ago? | Manifest + prompt history + model snapshot (if still available) |

**The uncomfortable truth to state:** *"You can only reproduce an output if the model snapshot is still available from the provider. Most providers deprecate snapshots on 6–12 month cycles. So for anything with a regulatory retention requirement longer than that, either self-host the pinned weights, or accept that you can reconstruct the *inputs* but not the *output*. Say which one you have chosen — regulators will ask."*

### 10.3 Regulatory touchpoints

| Regime | Release-process implication |
|---|---|
| **EU AI Act (high-risk)** | Documented risk management per change; logging; human oversight; technical documentation per release |
| **SOC 2** | Change management evidence; segregation of duties; approval trails |
| **ISO 42001** | AI system lifecycle documentation; impact assessment |
| **GDPR / deletion** | A deletion request must reach the index, the cache, and any fine-tuning data — a release process must handle all three |
| **Financial services** | Model risk management; independent validation before release |

**The GDPR point that trips people up:** *"A deletion request cannot reach into model weights. So the release process must record, per release, which training data went into it — otherwise you cannot answer a deletion request that touches a fine-tuned model. Either the model was never trained on user data, or you need a documented retraining path."*

---

## 11. Interview Q&A

**Q1. What's fundamentally different about CI/CD for an LLM system?**
Four things. The artefact is not one thing — it's four versions that couple and must be released together: code, prompt, model, index. So I introduce a release manifest with a single release ID that pins all four. Tests are statistical rather than binary, so gates are thresholds on metrics, not pass/fail assertions. Failures are quiet — a prompt change that degrades quality produces no errors, no latency change, nothing a classical monitor catches, so I need quality signals in the deploy gate itself. And rollback must restore a combination of versions, which means it has to be a pointer flip rather than a redeploy.

**Q2. Walk me through the pipeline.**
Six blocking gates. Static validation for syntax and token budgets, unit tests for determinism and parsing, a golden set for obvious regressions, a sampled LLM-judge stage for quality, a safety stage for injection and PII, and a cost and latency check. All blocking, about 25 minutes total, roughly $59 per release. Then deployment in three stages: shadow first, where I mirror production traffic and compare outputs offline without affecting users; then canary at 5% with a 30-minute bake stepping up through 25, 50, 100; then a 24-hour watch window where on-call owns the change.

**Q3. Why shadow before canary?**
Because a canary tells you whether the new version breaks, and shadow tells you whether it's *different*. For a model change or a system prompt change, the outputs differ in kind — a different refusal rate, different verbosity, different failure modes — and comparing error rates at 5% traffic won't surface that. Shadow gives me a side-by-side on the same inputs, so I can measure divergence rate, quality delta, and safety delta without any user impact. I shadow every change that can alter the kind of answer, and I skip shadow for changes that can only alter the speed of the answer.

**Q4. What are your rollback triggers?**
Error rate above 5% for 5 minutes. p95 latency above 1.5x the production baseline for 10 minutes. Quality score more than 0.05 below the production baseline in the canary. Cost per request above 1.5x for 30 minutes — that one pauses rather than rolls back. And any safety violation at all — that's zero tolerance with no bake time. The quality trigger is the one people forget, and it's the one that matters, because that's the failure mode nothing else catches.

**Q5. How fast is your rollback, and how?**
Under 5 minutes, typically under 90 seconds. It's a pointer flip in the release registry — nothing is rebuilt, nothing is restarted, the router just reads a different pointer. That's the whole reason to have a release registry. The prerequisite, which is the part people miss, is that the previous release's artefacts must still be servable: the image cached, the prompt bundle present, the model snapshot still available from the provider, and critically, the previous index version still queryable. If you rebuilt the vector index in place, you cannot roll back retrieval at all — the rollback restores the old prompt but points at a new index, and you get a hybrid state that never existed in testing. So indexes are immutable versioned collections and the deploy is an alias flip.

**Q6. How do you test a prompt change?**
Four stages, and the key insight is that outputs are non-deterministic so testing must be statistical. Static validation catches syntax and token-budget problems in seconds. Unit tests catch template rendering and output-parsing breakage. The golden set — a couple of hundred fixed input/output pairs, some exact-match and some must-contain — catches obvious regressions with no judge cost. And then sampled LLM-judge evaluation on a larger set, maybe a thousand cases at 20%, measuring relevance, accuracy and faithfulness. All of it gates. But the most important thing I do is read the *diff* of the eval report, not the score — per-slice deltas and the top ten changed cases. A two-point aggregate gain that hides a fifteen-point loss on the multi-hop slice is a regression wearing a good number.

**Q7. How do you deploy a model version change?**
More slowly than anything else, because you can't diff a model. I pin the snapshot — never a floating alias. I confirm the tokeniser, because a tokeniser change silently changes every token count and therefore every cost and every truncation boundary. I run the full eval suite offline, not sampled, plus a regression set of everything that ever failed, plus an adversarial set, plus a format-compliance check on tool-call JSON, plus a long-context check. Then 72 hours of shadow rather than 24, because model differences surface slowly. Then canary with 2-hour bakes rather than 30 minutes. And I keep the old snapshot warm for 7 days, because rollback has to stay possible for the full window.

**Q8. What's the cache-key bug that bites people?**
If the semantic cache key is just the query hash, then after a release users get old answers from the previous release for up to the TTL. So you deploy the fix and the bug is still happening, and on-call spends 40 minutes debugging the prompt before finding it. The cache key must include the release ID. The cost is that every release starts with a cold cache, which is a temporary latency and cost spike — so I pre-warm the top thousand queries against the new release before flipping the pointer, and I create an alert for cache hit rate dropping below 10% that is explicitly marked as expected after a deploy, so nobody pages on it.

**Q9. What does the pipeline cost?**
About $59 per release, of which $48 is shadow on 10,000 requests. At 12 releases a day that's about $21,000 a month. Compare that to one bad release caught late: a full rollout with an 8-minute MTTR means roughly 4,000 affected requests plus 8 engineer-hours plus the trust cost. One prevented bad release pays for about three weeks of pipeline. The gate is the cheapest component in the system, and I'd say that explicitly because the instinct is to see $21k a month as expensive.

**Q10. How do you keep the eval set honest?**
It's versioned infrastructure with the same rigour as the prompt. Changes to it are their own PR, reviewed by someone other than the author, and never ship in the same release as the change they're measuring. Adding cases is fine. Removing cases needs a written reason and a second reviewer. And changing expected outputs needs a second reviewer specifically because that's the mechanism by which a regression gets laundered into a passing test. The rule I hold to is: if I'm ever tempted to adjust the eval set to make a failing release pass, that is precisely the signal to roll back instead.

**Q11. How do you handle a release during an incident?**
It's frozen automatically. If an incident is open, only rollback targets are permitted — because adding a variable while you're troubleshooting is how a one-hour incident becomes a four-hour one. That's enforced in the release tooling, not by convention. The only exception is a hotfix that *is* the incident fix, and that requires the incident commander's recorded approval and a minimal pipeline of static plus golden plus safety gates. Every emergency release goes through the full pipeline retrospectively within 48 hours, and if it fails, it gets reverted.

**Q12 (pushback). This process takes 27 hours. Isn't that far too slow?**
For a high-risk change, yes it's slow, and that's deliberate — I tier by risk. A config tweak gets a 2-hour shadow and a 30-minute canary, which is about 4 hours end to end. A prompt-template change gets 24 hours of shadow. Only system prompts and model versions get the full treatment, and those are changes I make maybe monthly, not daily. So the median release is a few hours and the tail is a day. And the reason I'm willing to pay the tail is that the failure mode I'm protecting against is silent: a system prompt change that costs me eight points of faithfulness produces no alerts, no errors, and no latency change. The process is the only thing that catches it.

**Q13. How do you attribute a quality drop if multiple things shipped?**
That's exactly why I use a release train rather than continuous deployment. All changes for a release window are batched into one manifest with one release ID, and that release is the unit of attribution. If quality drops, I know it's release N and I can bisect within it offline — rerun the eval with each component reverted. If I deployed continuously, three prompt changes and a model swap landing in an hour would be unattributable. The cost is lead time, which is why the train runs twice a day and hotfixes bypass it.

**Q14. What do you log per request for auditability?**
The `release_id`, and from that the entire manifest: git SHA and image digest, prompt bundle version, model ID and pinned snapshot, index version, guardrail policy version, and all the config. Plus the trace ID, the retrieved document IDs, the tool calls, the token counts, and the cache status. So when someone asks "which model version generated this response on the 14th of March and with what prompt", I can answer from a single log line. The uncomfortable caveat is that full reproducibility also requires the model snapshot to still exist — most providers deprecate on a 6-to-12-month cycle — so for anything with a longer retention requirement I either self-host the weights or I'm explicit that I can reconstruct the inputs but not the output.

**Q15. What's the one thing teams get wrong most often?**
Treating prompts as configuration rather than as code. A prompt edited directly in a production config map is an unreviewed, unversioned, unevaluated release that changes 100% of system behaviour. The second most common is rebuilding the vector index in place, which silently destroys your ability to roll back retrieval. And the third is not putting the release ID in the cache key. All three have the same shape: a change that looks like a small operational convenience and is actually an uncontrolled release.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Release artefact** | Manifest pinning **code + prompt + model + index** |
| **Release ID** | Single pointer that identifies the whole combination |
| **Model pinning** | Snapshot ID, **never a floating alias** |
| **Index versioning** | Immutable collection + alias flip, never rebuild in place |
| **Gates** | G1 static · G2 unit · G3 golden · G4 quality · G5 safety · G6 cost |
| **Gate time** | ~25 min total |
| **Gate cost** | ~$59/release ($48 shadow) · ~$21k/month at 12/day |
| **Golden pass threshold** | ≥ 0.95 |
| **Quality thresholds** | relevance ≥ 4.0 · accuracy ≥ 4.0 · helpfulness ≥ 3.5 |
| **Faithfulness SLO** | ≥ 0.85 |
| **Safety thresholds** | Injection/PII/unauth action = **0, zero tolerance** |
| **Shadow** | 24 h or 10k requests · **72 h for model changes** |
| **Shadow divergence limit** | < 10% without quality gain |
| **Canary** | 5% → 25% → 50% → 100% · 30 min bake (2 h for models) |
| **Auto-rollback triggers** | error > 5%/5m · p95 > 1.5x/10m · quality −0.05 · **any safety violation** |
| **Rollback** | Pointer flip, **< 90 s**, target < 5 min |
| **Releases kept hot** | 10 |
| **Model snapshot retention** | ≥ 7 days before decommission |
| **Old index retention** | 14 days |
| **Watch window** | 24 h, on-call owns it |
| **Release train** | 2× daily; hotfixes bypass |
| **Change failure rate SLO** | < 10% |
| **MTTR SLO** | p50 < 5 min, p95 < 10 min |
| **Lead time SLO** | p50 < 45 min, p95 < 3 h |
| **Gate flake SLO** | < 3% (7-day) |
| **Four-eyes** | System prompt · guardrail policy · eval expected outputs |
| **Cache key** | **MUST include `release_id`** |
| **Freeze rule** | incident open → rollback targets only |
| **Budget rule** | > 60% monthly error budget consumed → feature freeze |
| **Record retention** | Release registry 7 y · gate results 2 y · traces 90 d hot |
| **The rule that never breaks** | **Only the pointer changes at deploy time — nothing is rebuilt.** |

---

## 13. Cross-links

- [Capacity Planning and Autoscaling](05-capacity-planning-and-autoscaling.md) — sizing for shadow headroom and canary capacity
- [Cost and FinOps](06-cost-and-finops.md) — cost regression detection and per-release cost attribution
- [Reliability Patterns](07-reliability-patterns.md) — circuit breakers, retries, and degradation ladders
- [Security, Guardrails and Compliance](08-security-guardrails-and-compliance.md) — the guardrail policy release path
- [DR, Failover and Backup](09-dr-failover-and-backup.md) — releasing into a failover region
- [LLMOps for AI Systems](10-llmops-for-ai-systems.md) — the broader operational lifecycle
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — the eval set and judge design these gates use
- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md) — the index versioning this pipeline depends on
- [LLM Serving, Inference and Gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) — the router that reads the release pointer
- [Fine-Tuning vs RAG vs Prompting](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) — adapter versioning and base-model pinning
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
