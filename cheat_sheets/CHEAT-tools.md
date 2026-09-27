# Cheat Sheet · Choosing Eval Tooling

> **Covers:** CS-04 (the complete workflow), CS-19 (building vs buying), CS-22 (scaling evals) ·
> **Code ground truth:** all seven dossiers, `CODE-01` … `CODE-07` · **Use when:** someone asks
> "which eval framework should we use?" and you want to answer that question correctly instead of
> naming the one you have heard of.

---

## The 60-second version

There is no best framework, because the eval stack has **five distinct layers** and different tools
own different layers. Answering "which tool" before naming the layer produces the wrong answer every
time.

| Layer | The job | Pick by |
|---|---|---|
| **Process** | the order of operations | team discipline, not tooling |
| **Runner** | execute evals, resolve datasets, collect metrics | breadth vs depth, and language fit |
| **Judge** | score subjective criteria | validate before you compare — the *validator* matters more than the vendor |
| **Benchmark adapters** | published capability numbers | coverage, and whether you need multimodal/agent |
| **Platform** | traces + scores + datasets + experiments in one store | whether your problem is production or pre-release |

Two facts that decide most choices: **if your problem is production quality, you need a platform**
(a trace store where a score sits next to the latency and cost of the same interaction).
**If your question is "can it actually do the task", you need execution grading**, which almost
nothing does.

---

## Core concepts

| Concept | What it means | Where it shows up |
|---|---|---|
| **The registry pattern** | a name resolves to an implementation (a `class:` string, a path, a config) so the core never changes when you add | 4 of the 7 dossiers are registries |
| **Persistence before scoring** | write the prediction before review begins | so re-judging is cheap and a crash never costs a generation |
| **Harness confound** | same model, different scaffolding → different ranking | the leaderboard must name the harness in the same column as the model |
| **Execution vs output grading** | run the artefact, or read its prose | the single biggest design axis in the collection |
| **Online ↔ offline duality** | the *same* evaluator on sampled production and on a frozen dataset | the reason offline numbers predict online behaviour |
| **Harness self-test** | a mock model / dummy solver that exercises the pipeline with no spend | validate your plumbing before you pay for a rollout |
| **Deterministic sampling** | the same trace is always scored the same way across retries | a naive `Math.random()` sampler gets this wrong |

---

## The selection matrix

| Tool | Layer | Stack | Best at | Choose it when | Do **not** use it for |
|---|---|---|---|---|---|
| **`CODE-01` awesome-evals** | Process / canon | Markdown | The judge recipe, pass@k vs pass^k, error analysis, the pattern playbook | You need to write or *audit* an evaluator and want the canonical recipe with code | Running anything — it is not executable |
| **`CODE-02` OpenAI evals** | Runner (judge) | Python + YAML | Zero-code custom evals — a judge is a YAML prompt + `choice_scores`; an eval is 3 keys | Your criterion is subjective and custom to your product | Benchmarks, production monitoring, execution grading |
| **`CODE-03` evals-skills** | Process | Markdown / agent skills | Process enforcement: error analysis *before* evaluator writing; validate before trust | Your failure mode is process drift, not missing tooling | Being your only infrastructure — it ships no runner |
| **`CODE-04` Evalscope** | Runner + benchmarks | Python + TS | Breadth: **205 benchmark adapters**, multimodal, agent mode with a Docker sandbox, TTFT/TPOT perf | You need published benchmark numbers, or quality **and** latency | Long-horizon execution grading |
| **`CODE-05` frontier-evals** | Execution grading | Python + Docker | Running the artefact: rollout → reproduce in a fresh GPU container → rubric-grade in a third | The question is "can it actually do the research/engineering task" | Product evals, fast iteration, anything needing a score in minutes |
| **`CODE-06` Langfuse** | Platform | TypeScript | Traces, scores, datasets, experiments, annotation queues in one store; online and offline share evaluators | Your problem is **production** quality and you have real users | Benchmarks, model selection, execution grading |
| **`CODE-07` search_evals** | Runner (hosted systems) | Python (async) | Cost-accounted hosted-agent evals; per-component Decimal cost; agent vs grader split; resumable runs | You are evaluating a hosted deep-research/agent product you do not control | Self-hosted models, offline batch scoring |

**A realistic stack, assembled:** `CODE-03` to find your failure modes → `CODE-01` to write and
validate the judge → `CODE-02` (or a code evaluator) to implement it → `CODE-06` to run it offline on
datasets and online on sampled production traces → `CODE-04` when you need to choose or
regression-test the model underneath.

---

## Decision rules

1. **Name the layer before naming the tool.** "Which framework" is four questions wearing one coat.
2. **Process before tooling.** If your team writes evaluators before doing error analysis, no
   framework fixes that; adopt the order of operations first.
3. **Code check before judge, always.** If the property is checkable in code, a judge is strictly
   worse — slower, costlier, noisier, and not unit-testable.
4. **The validator matters more than the vendor.** Any judge is only as good as its TPR/TNR on your
   data. Compare frameworks on how easily you can *validate*, not on metric count.
5. **Choose execution grading only if the task is genuinely executable.** It costs GPU containers and
   hours of wall-clock; if "does it sound right" is the actual question, it is the wrong tool.
6. **Buy a platform when a second person needs to look at traces.** Before that, scripts and a
   spreadsheet are correct; after that, a trace store is not optional.
7. **Prefer a registry-shaped tool.** Extensibility without core changes is the property that
   determines whether the tool survives your second use case.
8. **Check the harness before you believe the score.** Report which scaffolding produced it; the same
   model can rank differently under a different harness.
9. **Insist on persistence-before-scoring.** Re-judging must be cheap, and a crash must not cost you
   a generation.
10. **Test the harness itself with a mock.** A mock model or dummy solver exercises the full pipeline
    for zero spend — copy this into anything you build.
11. **Self-host only if you will actually operate it.** Five stateful components is a real cost; the
    data-plane freedom is worth it only if you have someone to run it.
12. **Do not adopt a tool for a metric you have not validated in it.** "It ships a faithfulness
    metric" is not the same as "its faithfulness metric agrees with your humans".

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Benchmark adapters in Evalscope | **205** |
| PaperBench top score | **26.0 ± 0.3** (IterativeAgent o1-high, 36 h limit) |
| PaperBench Code-Dev (same agent) | **43.4 ± 0.8** — writing code is much easier than reproducing a result |
| Longest agent rollout observed | **36 hours**, in a GPU-backed container |
| Runs reported per leaderboard entry | **3**, with ± reported |
| Cost accounting in `CODE-07` | **Decimal**, per component, merged across agent and grader |
| Judge model in `CODE-07` | `gpt-4.1`, strict JSON schema with a `const true` field |
| Permanent (non-retryable) HTTP statuses | `{400, 401, 403, 404, 422}` |
| Mocking hooks to look for | `mockllm.py` (`CODE-04`), `dummy` solver (`CODE-05`) |

---

## Tool commands

```bash
# --- Codebase dossiers: reading order if you are new to all seven ---
#  1. CODE-01 PATTERNS.md § LLM-as-judge      (20 min, highest value page)
#  2. CODE-03                                 (the order of operations)
#  3. CODE-02                                 (how cheap a real evaluator is to declare)
#  4. CODE-06                                 (where it all has to live to be useful)
#  5. CODE-07                                 (cost as a first-class metric)
#  6. CODE-04                                 (benchmark vs eval)
#  7. CODE-05                                 (the ceiling of execution grading)

# --- Evalscope (CODE-04) ---
#   benchmarks/_index.json  maps a benchmark name -> adapter path  <- the registry
#   models backends include mockllm.py -> test the harness for free

# --- frontier-evals (CODE-05): cheapest honest path is the dummy solver ---
cd project/paperbench && uv sync
uv run python -m paperbench.grade --help       # stage 3
uv run python -m paperbench.reproduce --help   # stage 2
#   read solvers/dummy first: it exercises all three stages with NO model spend

# --- Langfuse (CODE-06) ---
#   online path : trace -> traceFilterUtils + decisionModel -> deterministicSampling -> judge -> score
#   offline path: dataset -> experiment -> trace + scores per item -> diff configurations
```

---

## Top 10 mistakes

1. **Choosing a tool before naming the layer** it is supposed to fill.
2. **Adopting a benchmark runner for a production problem** (or vice versa).
3. **Reaching for execution grading** when "does it sound right" was the question.
4. **Comparing frameworks on metric count** instead of on how easily you can validate their judges.
5. **Trusting a shipped metric** without checking its TPR/TNR on your own labels.
6. 6. **Ignoring the harness** when reading a leaderboard — the scaffold can flip the ranking.
7. **Self-hosting a five-component platform** with nobody to operate it.
8. **No mock/dummy self-test**, so the first real run is also the first test of the plumbing.
9. **Non-deterministic online sampling**, making retries incomparable.
10. **Building a bespoke runner** when a registry-shaped framework already covers your case — and the
    inverse: adopting a heavyweight platform when a script and a spreadsheet would do.

---

## If you only remember three things

1. **Name the layer first** — process, runner, judge, benchmark adapters, platform. "Which framework"
   is not one question.
2. **Code check before judge; validate before trust; report the harness alongside the score.**
3. **Four of the seven tools are registries, and two ship a mock — because extensibility without core
   changes and a free way to test your own plumbing are what make an eval stack survive.**
