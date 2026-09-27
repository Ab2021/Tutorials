# Interview Pack · Benchmarks, Leaderboards & Model Selection

> **Covers:** CS-09 (reading a leaderboard), CS-10 (the evolution of knowledge benchmarks), CS-11
> (saturation vs contamination), CS-12 (selecting the right LLM — the three-stage selection, the
> cost-filter arithmetic, and execution-based custom model evals) ·
> **Code ground truth:** `CODE-02` (openai/evals — an eval is YAML plus optional code), `CODE-04`
> (EvalScope — 205 benchmark adapters behind one CLI) · **Role level:** mid → senior ·
> **Format:** screen → onsite → model-selection case

---

## How this domain is assessed

Benchmark literacy is the cheapest strong signal in an LLM-evals interview, because almost everyone
can name three benchmarks and almost nobody can say what one *is*. The candidate who defines a
benchmark as **four components** and then refuses to read a bare number is a different hire from the
one who quotes leaderboard order.

The interviewer is testing one thing above all: **do you treat a score as evidence or as a fact?**
Everything in this pack follows from that. A strong candidate is not cynical about benchmarks — they
can name what each one was built to measure, why it worked, and how it will die.

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Definitional precision** | Do you know a benchmark from a leaderboard, and its four parts? | "Define a benchmark." |
| **Disclosure instinct** | Do you ask for the configuration before believing the number? | "This model scored 92%." |
| **Saturation literacy** | Do you know a high score can mean a dead benchmark? | "Top ten are all 92–94. Good?" |
| **Contamination awareness** | Do you know public + pretrained = suspect? | "How do you know it didn't memorise it?" |
| **Selection discipline** | Do you filter then evaluate, or just pick the top row? | "Which model should we use?" |
| **Historical fluency** | Can you explain *why* each benchmark exists? | "Why was MMLU-Pro created?" |

**The single most predictive question in this pack** is *"what would you need to know before you
believed this score?"* A strong candidate produces the disclosure list from memory — configuration,
scoring strategy, aggregation, dataset age, who ran it, confidence interval, composite weights.

---

## Tier 1 — Fundamentals (10 Q)

### Q1. What is a benchmark, precisely?

**What they're really testing:** whether you have a definition or a vibe.

**Model answer:**
"A **standardised test used to measure a particular model capability**." And the standardisation is
structural — every benchmark has exactly **four components**, and all four are documented in its
research paper:

| # | Component | What it covers |
|---|---|---|
| 1 | **Dataset + task** | the questions **and** their answers, plus the instruction of what the model must do. The source's own phrase is "like a **golden dataset**" |
| 2 | **Run configuration** | prompt construction (zero-shot vs few-shot, chain-of-thought on/off), decoding (temperature, max tokens), scoring strategy (pass@1 / pass@k / majority@k), and tool access |
| 3 | **Scoring method** | **extraction** of the answer from free-form output, then comparison against ground truth |
| 4 | **Aggregation method** | how per-question scores become one headline number |

The reason the definition matters is that **a number is meaningless unless all four are known and
identical across the models being compared.** When you see "92% on MMLU", you have been told
component 4 and nothing else.

**Red flags:** "it's a test for LLMs"; describing only the dataset; no mention of configuration;
treating the score as the benchmark.

### Q2. What is the difference between a benchmark and a leaderboard?

**What they're really testing:** whether you know where the trust is supposed to come from.

**Model answer:**
- A **benchmark** is the exam — "an exam that tests LLMs on one particular aspect."
- A **leaderboard** is "a public ranking and comparison table that shows how different LLMs perform
  on a common set of evaluations." The benchmark is the exam; the leaderboard is the notice board.
- The leaderboard's value is that it makes comparison *possible* — a score for one model is not
  comparable to another's unless both sat the same exam under the same conditions, and the leaderboard
  is the artefact that asserts that.
- **Four types, and they are not equally useful:**
  - **Benchmark-specific** — one exam only (Humanity's Last Exam, where a frontier model sat at 38%).
    Least useful for selection, most useful for tracking one capability.
  - **Multi-benchmark / composite** — several benchmarks joined into one score (LiveBench: 23
    objective tasks across 7 categories; Artificial Analysis). Most useful, and usually the only kind
    that also carries **cost, latency, speed and context** alongside quality.
  - **Human-preference** — pairwise votes on anonymous answers (LMArena). Biased toward long,
    confident, well-formatted answers.
  - **Application-specific** — one domain, many benchmarks (BFCL for function calling, MTEB for RAG
    embeddings). The right board when you know your use case.

**Red flags:** treating them as synonyms; not knowing that composite boards hide their weights; no
awareness that human-preference boards carry human bias.

### Q3. Why do leaderboards exist? What do they actually buy you?

**What they're really testing:** whether you can state the value before you criticise it.

**Model answer:**
Five things, and a critique only makes sense against them:
1. **Comparison across labs on a common reference** — the only way to compare two vendors' models
   without running both yourself.
2. **Third-party trust** — a lab-neutral evaluator's number is more credible than a lab's claim about
   its own model.
3. **Model selection at a scale you cannot afford to do yourself** — you cannot evaluate 100 models;
   you can read a board.
4. **Saturation detection** — when the top 10 models on a knowledge benchmark all sit between **92 and
   94**, the benchmark has stopped discriminating. That is useful information the board gives you for
   free.
5. **Discovery below the top three** — models at ranks 10, 12, 15 and 20 are "generally cheaper" and
   often sufficient.

**Six stakeholder groups, and they want different things from the same board:** AI engineers
(shortlisting), frontier labs (release strategy), researchers (research direction), policy makers and
safety institutes (monitoring), and the open-source community (discovery and publicity). A candidate
who names two of those has read a leaderboard; one who names all six has thought about who is using it.

**Red flags:** describing leaderboards as marketing only; no awareness that the same board serves
contradictory purposes.

### Q4. Walk the lifecycle of a knowledge benchmark.

**What they're really testing:** whether you see benchmarks as a lineage rather than a list.

**Model answer:**
Seven benchmarks, one story — each built to fix the previous one's failure, each destined to saturate:

| # | Benchmark | Year | What it opened |
|---|---|---|---|
| 1 | **MMLU** | Sept 2020 | **Breadth** — 57 subjects, 14,000 4-option MCQs. The "mother of all benchmarks" |
| 2 | **TruthfulQA** | Sept 2021 | **Reliability** — 817 adversarial questions in 38 categories; asking whether the model is truthful, not merely knowledgeable |
| 3 | **AGIEval** | April 2023 | Test on **existing human exams** instead of inventing new ones — 20 real exams, 8,000+ questions |
| 4 | **GPQA** | Nov 2023 | **Depth** — physics/chemistry/biology questions that a non-specialist cannot solve in 30 minutes with Google |
| 5 | **MMLU-Pro** | 2024 | **Repair MMLU** — 4 options → 10, rote → reasoning, 57 subjects → 14 balanced categories |
| 6 | **SimpleQA** | 2024 | **Calibration and hallucination** — 4,326 short-answer factual questions with no options given |
| 7 | **HLE** | Jan 2025 | **Breadth × depth in one** — 2,500 questions, 100+ subjects, 1,000 experts, a **private test set**, and calibration measured |

The recurring mechanism, stated once and demonstrated seven times: **a benchmark's questions are
public, they enter the next generation's training data, models converge, and the benchmark
saturates.** Once frontier models cluster, it cannot discriminate, and something new is built.

Two secondary threads worth naming: **marketing distortion** — every headline ("an LLM beat humans in
JEE/NEET", "we beat PhDs on GPQA") is narrower than it sounds — and the fact that **a benchmark being
publicly available means anyone can evaluate any model**, which raises the question of whose
evaluation you should believe.

**Red flags:** naming benchmarks without the causal chain; not knowing that TruthfulQA found **bigger
models were *less* truthful**; treating the list as a ranking of importance.

### Q5. What is knowledge capability, and what is the breadth/depth distinction?

**What they're really testing:** whether you know what these benchmarks actually measure — and what
they do not.

**Model answer:**
- **Knowledge capability is the model's *parametric* world knowledge** — how much it retained from
  training, as distinguished from anything it retrieves. The test is "what is hidden in the
  parameters."
- It is the **most fundamental** capability historically: when the first LLMs were trained on
  internet-scale data, the expectation was simply that you could ask about anything on the internet.
  Reasoning, emergent behaviour and coding came later, with scale.
- **Breadth vs depth** is the organising pair: you rarely find both in one person, because depth costs
  time. **MMLU tests breadth** — 57 subjects at a basic level. **GPQA tests depth** — three subjects
  at research level. **HLE tests both at once.**
- **Crucially, a knowledge score is not a reasoning score, and neither is an agentic score.** A
  candidate should say explicitly that these benchmarks do not predict open-ended or agentic
  performance — a model can score in the frontier band on MMLU and behave entirely differently on a
  task that requires tool use, ambiguity handling, or a long horizon.

**Red flags:** conflating knowledge with intelligence; claiming a knowledge benchmark predicts agent
performance; no awareness that "parametric" is the point.

### Q6. What is benchmark saturation, and how do you detect it?

**What they're really testing:** whether you know a high score can be bad news.

**Model answer:**
- **Saturation is when all models cluster at the same high score, so the benchmark can no longer
  differentiate them.** The lecture's illustrative cluster is **95 / 94 / 92**; the equivalent
  formulation is "the top 10 models all between **92 and 94**."
- The **progression** is the tell: 25% → 36% → 50% → 70% → **90–95%**. The interesting work happens in
  the middle. Once everyone is compressed into the top band, the benchmark has stopped measuring.
- The mechanism is always the same: a **static benchmark** (the dataset published in the paper five
  years ago is the same dataset today) against **improving models**.
- **Two reasons a benchmark becomes obsolete, and both are usually present:** it is available on the
  web, and it entered training data so the model has memorised it.
- Detection is therefore partly statistical (variance across frontier models collapsed) and partly
  mechanical (check the invalid-question rate — **~6.5–8% of MMLU's questions are simply wrong**,
  which caps every model near 92% regardless of capability).
- The right response is not cynicism. **A benchmark is a perishable asset** — it saturates usually
  within one to two years and sometimes within two months — so benchmark construction has to be a
  **continuous practice** rather than a one-off project.

**Red flags:** treating saturation as a data-quality problem; not checking the invalid-question rate
before declaring a capability ceiling; quoting a saturated benchmark because it is familiar.

### Q7. What is contamination, and how do you detect it?

**What they're really testing:** whether you know memorisation from capability when you see it.

**Model answer:**
- **Contamination is the benchmark's questions and answers having become part of the model's
  pretraining data**, so you cannot tell thinking from memorising.
- The **condition** is precise, and worth stating as a conjunction: the dataset is **public** *and*
  the model's pretraining included a **scrape of the public internet containing it**. Both terms
  matter — a public dataset that no crawl reached is not contaminated by that crawl.
- **Detection approaches, in order of practicality:**
  1. **Canary strings** — a marker injected into the dataset, so if the string appears in a model's
     answer it is visible that the dataset was consumed during training. The honest caveat is that it
     is **mostly not possible** to predict in advance whether a dataset entered a model's training
     process.
  2. **Score-shape forensics** — a model that is perfect on the easy tail and the hard tail equally, or
     that answers an obviously corrupted question correctly, is memorising rather than reasoning.
  3. **Fresh or private test sets** — HLE's private held-out set is the structural answer; MMLU-Pro's
     regeneration is the repair answer.
- **One subtlety candidates miss: ask *which training stage* leaked.** Pretraining filters do not catch
  **alignment-stage** contamination, so a reliability benchmark can be contaminated by the alignment
  data even when the pretraining pipeline was clean.
- And the honest note: contamination **inflates a score without improving the model**, which is why it
  belongs in the same list as saturation, configuration gaming and aggregation hiding — four separate
  reasons a number is untrustworthy.

**Red flags:** "we filtered the training data" as a complete answer; no awareness of alignment-stage
contamination; treating contamination and saturation as the same phenomenon.

### Q8. pass@1, pass@k, majority@k — what is the difference and when does each apply?

**What they're really testing:** whether you know that a score is only comparable to one taken the
same way.

**Model answer:**
- **pass@1** — one showing, one answer, right or wrong. The default, and the only one that measures
  what a single deployed call would do.
- **pass@k** — the same question asked *k* times; the item is correct if **at least one** attempt is
  right. The source calls it "a more lenient strategy," and it measures *capability with retries*:
  the right metric when best-of-k is a legitimate deployment strategy, or when the question is "can
  this model do this at all."
- **majority@k** — the same question asked *k* times; the **mode** of the answers is taken as the
  answer. This measures *aggregated reliability*: it rewards a model that is usually right over one
  that is occasionally right. It is the natural metric for self-consistency-style reasoning.
- The interview-relevant point is not the definitions but the **comparability rule**: comparing a
  pass@k number against a pass@1 number is not a comparison. It is a configuration difference that can
  move a score by more than the gap between two ranks.
- And the cost consequence: pass@k and majority@k multiply your inference bill by *k*. A headline that
  used majority@k at k=8 is reporting a system that costs eight times as much per query, and the board
  may not say so.

**Red flags:** treating them as equivalent; not asking which was used; comparing them across models;
no awareness of the cost multiplier.

### Q9. What is extraction, and why does it matter more than people think?

**What they're really testing:** whether you know that scoring is a pipeline, not a comparison.

**Model answer:**
- **Extraction is pulling the answer out of free-form model output** — "the first step during
  scoring." It sits between generation and comparison, and it is where a surprising number of
  benchmark numbers are actually determined.
- The pipeline is: **generate → extract → compare against ground truth → aggregate.** The comparison
  is only as good as the extraction. If the agent's answer is correct but written as prose and your
  extractor expects a number, you score it zero.
- The comparison itself takes one of three forms, and they are not equivalent in cost or reliability:
  **programmatic** (exact match, execution, unit tests), **LLM-as-a-judge**, and **human**. The
  grading-method choice belongs in the disclosure alongside the configuration.
- Two operational facts that make extraction matter in practice:
  - **Extraction failure looks exactly like model failure.** A benchmark score that drops after a
    prompt change may be an extractor that no longer matches the output format. This is the most
    common silent regression in a benchmark harness.
  - **It is the step everyone skips.** "Skipping the extraction step and comparing raw output strings"
    is a named top-10 mistake — a correct answer with a trailing period scores zero against an exact
    match.
- Note that extraction is a **model-eval** concern; for application evals the equivalent problem is
  usually handled by structured output or a schema, which is one reason application pipelines are
  easier to grade honestly than public benchmarks.
- **CS-12's Text-to-SQL case is the best worked example of extraction mattering, and the lesson is
  "compare results, not strings."** Two SQL queries that look nothing alike can return the identical
  table; two that look nearly identical can return different tables. So the evaluator must:
  1. **Execute both** the model's SQL and the golden SQL against the real database.
  2. **Compare the result tables**, never the SQL text.
  3. Normalise before comparing — row counts, integer-versus-decimal types, and decimal precision.
  4. **Sort rows** — unless the query's semantics make ordering meaningful, in which case preserve it.
  The metric is then just `accuracy = correct / total` over the result-table comparison.
  The reason this is the canonical example: it shows that the grading method, not the model, is what
  makes a benchmark trustworthy. A string-comparison grader on this task would score a correct answer
  as wrong and would be trivially gameable by formatting; an execution-based grader is programmatic,
  reproducible, and **near-verifiable** — which is exactly the standard the judge literature asks for.
- The generalised rule: **whenever the output is executable or structurally checkable, execute it.**
  SQL, code, a JSON payload, a spreadsheet formula, a tool call — all of them are gradeable by
  running them, and all of them should be, before anyone reaches for a judge.

**Red flags:** treating scoring as a single step; no extractor versioning; blaming the model for an
extraction regression.

### Q10. What is aggregation, and why is weighting not a detail?

**What they're really testing:** whether you have ever actually computed a benchmark score.

**Model answer:**
- **Aggregation is how per-question scores become one headline number.** Two forms:
```
simple    score = correct_items / total_items         # 920/1000 = 92%
weighted  score = SUM(n_s * score_s) / SUM(n_s)       # over subjects s
          NEVER  SUM(score_s) / 57
```
- The per-item score is binary: `1` if the extracted prediction matches ground truth, `0` otherwise —
  "anything else is wrong, otherwise zero." Every headline number is a mean of ones and zeros.
- **Weighting matters because subjects are uneven.** MMLU has **57 subjects**, and they do not contain
  equal numbers of questions. An unweighted mean over subject percentages over-weights the small
  subjects. The correct computation weights by question count.
- **The subtler point is that a headline average hides the shape.** A 57-subject average of 92% is
  compatible with a model that is 99% on everything except one subject where it is 40%. That is
  **aggregation hiding**, one of the four named reasons a number is untrustworthy, and the fix is
  disclosure: publish per-subject scores, not only the average.
- Corollary for interpreting a board: **when two models are 0.2 points apart, the aggregation choice
  can be the entire difference.** Rank bias — reading 84.3 at rank 3 versus 84.1 at rank 5 as a real
  ordering — is the same error in a different costume.

**Red flags:** averaging percentages without weighting; quoting only the headline; no per-subject
data; treating a 0.2-point gap as a real difference.

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. You must pick an LLM for a RAG product. Walk the procedure.

**What they're really testing:** whether you filter and then evaluate, or just read the top row.

**Model answer:**
The five-step procedure, in order, and the order is the point:

1. **Write your own constraints first, before opening a leaderboard.** Application type, latency need,
   cost ceiling, context need, and deployment constraint (public API vs on-premise). A board cannot
   answer a question you have not asked.
2. **Choose the board that matches your work.** A RAG product wants **MTEB** for the embedding model,
   and a general capability board for the generator; an agent product wants a tool-calling board
   (**BFCL**) or an agentic board; a budget-sensitive product wants a composite board that carries
   cost and latency (Artificial Analysis).
3. **Read the definitions before the numbers**: what is scored, how, by whom, the inference budget,
   reasoning on/off, dataset age, update cadence, whether there is a private test set, whether it is
   saturated, whether confidence intervals are given, and the composite weights.
4. **Shortlist the top 3–5** models against your criteria — not the top 1.
5. **Run your own evaluation** on all of them against your own traffic, and pick from that.

The step people skip is 1, and it is the one that makes the other four coherent. The step people
resent is 5, and it is the only one that answers the actual question.

**CS-12 lands the same procedure as three explicit stages**, and the stage names are worth using
because they make the division of labour obvious:

| Stage | What you do | What it produces |
|---|---|---|
| **1. Requirements** | Write down your own constraints — capability needed, volume, latency ceiling, **budget** | A filter, not a wish list |
| **2. Leaderboard shortlist** | Apply the requirements to the boards to get **5–10** candidates | A shortlist |
| **3. Custom model eval** | Run those candidates on **your own data** with **your own** grading | The decision |

The worked example is a Text-to-SQL chatbot for a sports site — and note that CS-12 makes the point
that this is **explicitly not RAG** (the schema goes in the system prompt, so there is no retriever to
evaluate). The requirement stage produced a **₹3 lakh/month** budget; the leaderboard stage started
from **146 models**, computed the monthly cost of each from its own token estimates, and removed
everything above the budget — **50–60 models** gone before a single eval ran. The remaining
candidates were ranked with a **min–max normalised weighted score** of
`score = 9 × rating' + 1 × latency'` — a deliberate **90:10 capability-to-latency** weighting, with
the weights written down rather than implied. Top 10 by score, then **5** taken forward to the custom
eval.

The lesson to state: **the leaderboard did the cheap filtering, and the custom eval made the
decision.** Note also that the ranking was over *your* shortlist with *your* weights, not over the
board's ordering — which is the whole point of stage 2 being a filter.

**And the results are the punchline, because they invert the leaderboard.** On the custom eval the
spread was large and unexpected: the best model came in around **90%** accuracy on the held-out
queries, the next **85%**, then **80%**, then **65%**, and the last around **55%** — with **the model
that had been the week's headline release finishing last**. Three consequences worth naming:
- **Public rank did not predict task accuracy.** A model can lead a general board and be the worst
  choice for a schema-specific Text-to-SQL task. This is the single strongest argument against ever
  skipping stage 3.
- **The failure mode was visible in the artifacts, not just the score.** The weakest model was
  reported as producing **three to six SQL syntax errors** and running with default reasoning
  settings rather than tuned ones — which is what an execution-based grader surfaces and a string
  comparison would have muddled.
- **The final decision was not purely a score.** Between the top two candidates the lean was toward
  the one with better **API reliability and stability** rather than the highest accuracy, and the
  **PM — not the engineer — made the call**, with a vote if the room split. That is the right shape:
  capability is one input, operational stability is another, and the trade-off is a product decision
  that belongs to the person who owns the product.

**Red flags:** opening a leaderboard first; picking rank 1; skipping the custom eval; using a
general board for an embedding or tool-calling decision; shortlisting without a budget constraint
applied; weights chosen but unstated.

### Q12. Here is a leaderboard row. What is missing?

**What they're really testing:** whether you can produce the disclosure checklist from memory.

**Model answer:**
Given "Model X — 78.4", the missing set is:

- **Model identity**: exact version, and **quantization** — a different quantization is a different
  system.
- **Run configuration**: zero-shot or few-shot (and how many shots), chain-of-thought on or off,
  temperature, max tokens, tool access.
- **Scoring strategy**: pass@1, pass@k, or majority@k — and if k, what k.
- **Extraction method**: what turned the output into a comparable answer, and its version.
- **Grading method**: programmatic, judge, or human — and if a judge, which judge.
- **Aggregation method**: simple or weighted; and the **per-subject breakdown**, not only the average.
- **Who ran it**: a lab evaluating itself, or a third party.
- **Dataset provenance**: original or regenerated; how old; public or private; any known invalid
  questions.
- **Whether it is saturated**: what the other frontier models scored, so you can see the spread.
- **Confidence interval or standard error**: without it, a 0.2-point gap is meaningless.
- **Composite weights**, if this is a composite board — a hidden weighting is an undisclosed opinion.
- **Cost and latency for the same run**: quality in isolation is not a selection criterion.

The reason to produce this list at speed is that it converts a score from a **fact** into **evidence**,
and evidence can be weighed.

**Red flags:** accepting the number; asking only "which benchmark?"; no confidence interval; no cost.

### Q13. How much can run configuration change a score?

**What they're really testing:** whether you know the number is a function of the settings.

**Model answer:**
Enough to swamp every other signal, and there are three separate magnitudes worth knowing:
- **Scoring path, on the same model and benchmark: 84% versus 87% on MMLU** — by generation versus by
  log-probability. Three points, no capability difference.
- **Configuration gaming: a 5–10% swing.** The named pattern is a lab giving its own model the most
  favourable conditions while the rival gets the defaults. That is not a subtle effect — 5–10% is
  several rank positions on a saturated board.
- **Sampling strategy:** a pass@k or majority@k run at k = 8 is a different measurement from pass@1,
  and generally a higher number.
- The governing rule is a hard requirement, not a preference: **"it is not the case that you run one
  model in a particular setting and another model in a different setting — when you run both models,
  all the settings must be exactly the same."**
- Two defaults that follow: **temperature ≈ 0** is the comparability setting, and **max tokens must be
  high enough not to truncate chain-of-thought** — truncating the reasoning silently converts a
  reasoning failure into a configuration failure.
- The exam analogy worth using: the **benchmark is the exam paper**; the **eval harness is the entire
  examination administration**, handling everything behind the scenes so that every candidate sits
  under identical conditions.

**Red flags:** "the default settings are fine"; not asking about temperature; comparing a CoT run to a
non-CoT run; setting max tokens without checking truncation.

### Q14. How do you compare two models fairly?

**What they're really testing:** whether you can state comparability as a set of invariants.

**Model answer:**
A fair comparison holds everything constant except the model:
1. **Same dataset version** — including whether either run used a regenerated subset.
2. **Same run configuration** — prompt construction, shots, CoT, temperature, max tokens, tools.
3. **Same scoring strategy** — pass@1 against pass@1.
4. **Same extraction and grading method** — and the same **judge version**, if a judge is used. Judge
   drift means **judge-graded results should not be compared across years** at all.
5. **Same harness** — because the harness is a confound as large as the model.
6. **Same evaluation runner**: a lab's self-reported number and a third party's are different claims.
7. **Same quantization and version.**
8. **Report uncertainty**, and only then compare: a gap smaller than the confidence interval is not a
   gap.
9. **Compare price and latency at the same time**, because the deployment decision is a three-way
   trade.

And the honest failure mode: if you cannot establish all nine, **say the comparison is not available**
rather than reporting a ranking you cannot defend. "We don't know which is better on this axis" is a
respectable answer; a fabricated ordering is not.

**Red flags:** comparing self-reported to third-party numbers; no interval; comparing across judge
generations; treating a composite score as a like-for-like comparison.

### Q15. Why does benchmark performance not transfer to production?

**What they're really testing:** whether you understand what a benchmark deliberately removes.

**Model answer:**
The **Kaggle analogy** is the cleanest explanation available. There is a well-known claim that solving
Kaggle problems makes you a good data scientist in a real job, and the reason it was said is that
Kaggle data is very clean and the problem statement is very clear — so you get easy work. "In the real
world, everything is messy."

The parallel is exact: **the data in benchmarks is clean and models perform well on it.** Production
has:
- **ambiguous requests** and **missing information**,
- **company-specific data** the model has never seen,
- **tool failures** and partial outages,
- **unusual edge cases** and **long horizons**,

and "whether the model handles all of these — don't know."

Three further reasons benchmark performance is an upper bound rather than a prediction:
- **Contamination and over-optimisation** can inflate a score without improving the model — this is
  **Goodhart's Law**: "when a measure becomes a target, it becomes less useful as a measure." The
  car-mileage illustration is the one to use: if buyers only shop on mileage, the whole engineering
  team optimises mileage and the car gets worse in every other respect.
- **Saturation** means the benchmark may no longer be measuring the capability you care about.
- **The benchmark measured a model, and you are deploying a system.** The harness, prompts, retrieval
  and tooling are yours, and they can move the outcome more than the model choice did.

The practical conclusion is the whole point of the field: **a benchmark score shortlists; your own
evaluation decides.**

**Red flags:** "we'll just test the top model in staging and see"; no mention of contamination or
Goodhart; treating a public score as an SLO.

### Q16. MC1 versus MC2 — when does the choice matter?

**What they're really testing:** whether you have looked at how a multiple-choice benchmark is
actually scored.

**Model answer:**
- **MC1**: take the option with the highest log-probability (`argmax`) and treat it as the answer. It
  is a single-answer metric — exactly one option counts.
- **MC2**: for question *q*, sum the **normalized probability mass over all the true answers**, then
  take the mean over questions. It is the multi-answer metric, and it is **TruthfulQA's default**,
  because TruthfulQA questions are designed so that more than one option can be true.
- **The choice matters because it changes what "correct" means.** Under MC1, a model that assigns
  probability 0.9 to a true answer and 0.85 to another true answer is scored on the argmax alone.
  Under MC2, spreading mass across true answers is rewarded; spreading it across false ones is
  penalised.
- **The general lesson beyond multiple choice:** a benchmark's headline can be computed from
  probabilities rather than generated text, and that is a meaningfully different measurement. A
  log-probability score does not test whether the model would *say* the right answer — it tests
  whether the right answer is ranked highly. This is exactly the 84%-versus-87% MMLU divergence from
  Q13, and it is why "which scoring path" belongs in the disclosure.
- A related pattern worth naming: **SimpleQA's F-score is the harmonic mean of factuality and
  calibration** — `correct-given-attempted` excludes abstentions, so a model that abstains is not
  penalised on that term but is penalised on the other. Composite metrics like these encode a
  judgement about what matters, and the judgement should be read, not just the number.

**Red flags:** treating all MCQ scoring as equivalent; not knowing a benchmark can be scored from
log-probabilities; ignoring what a composite metric weights.

### Q17. When is an LLM judge acceptable inside a benchmark?

**What they're really testing:** whether you can place judge use on a spectrum rather than take a side.

**Model answer:**
Borrow the discipline from the judge literature and apply it to benchmark scoring:
- **First choice is always a programmatic check.** Extraction plus comparison against ground truth is
  cheapest, most reproducible, and ungameable. Anything decidable by equality should never call an
  LLM.
- **A judge is defensible for a binary question where domain-aware humans would agree on the answer**
  — what the field calls "pretty close to verifiable." Build robustness by **aggregating many binary
  criteria** rather than asking one holistic question, and prefer a cheap local model.
- **A judge is not defensible for fuzzy quality** — "is this a good answer?" — where there is no
  unanimity to appeal to. Find a proxy or accept that you are not measuring it.
- **Three benchmark-specific constraints on judges:**
  1. **Do not compare judge-graded results across years.** Judges drift, rubrics drift, and answer
     styles drift.
  2. **Disclose the judge and its version**, exactly as you would disclose the model.
  3. **Validate the judge** against a human-labelled held-out set — TPR and TNR — before it scores
     anything you will report.
- And the operational reason benchmarks have moved *away* from rubric judging: it is "very dependent
  on LLMs and their capability and very expensive," whereas action- or answer-level verification is
  cheaper and far more reproducible.

**Red flags:** defaulting to a judge; a judge with no validation; comparing judged scores across
years; a holistic 1–10 score where a binary criterion would do.

### Q18. What is calibration as a benchmark metric, and why did it need its own benchmark?

**What they're really testing:** whether you know what TruthfulQA and SimpleQA were built to catch.

**Model answer:**
- **Calibration asks whether the model knows that it does not know.** The definition in the source is
  exactly that: does the model know the answer is unknown to it.
- It needed its own benchmark because **capability benchmarks reward answering and never reward
  abstaining.** On an accuracy benchmark, a model that guesses has an expected gain; on a calibration
  benchmark, a confident wrong answer is the failure being measured. Scoring abstention as failure
  inverts the thing being tested — one of the named top-10 mistakes.
- The lineage: **TruthfulQA** (2021, 817 adversarial questions, 38 categories) established that
  knowledge and truthfulness are separate axes — and found the counter-intuitive result that **bigger
  models were *less* truthful**, because they reproduce internet misconceptions more fluently.
  **SimpleQA** (2024) is its successor: **4,326 short-answer factual questions that GPT-4 failed**,
  with no options given, scored on both factuality and calibration.
- **The metric design is the interesting part:**
```
correct-given-attempted = correct / attempted        # attempted excludes abstentions
F-score = 2 * (correct * cga) / (correct + cga)      # harmonic mean of the two
```
  The harmonic mean means **you cannot win by abstaining everywhere or by answering everything
  confidently.** Both degenerate strategies are penalised by construction. That is what a good
  composite metric does.
- **The number that makes the case for these benchmarks: the same model scores 88% on MMLU and 40% on
  SimpleQA.** A 48-point gap between "knows a lot" and "is reliable about what it knows."
- HLE carries the thread forward by measuring **calibration as the RMS between self-reported
  confidence and actual correctness** — so the newest benchmark treats confidence as a first-class
  output, not an afterthought.

**Red flags:** conflating calibration with accuracy; scoring abstention as failure; no awareness that
bigger models can be less truthful; treating hallucination as a knowledge problem only.

### Q19. How do you put cost and latency into a model-selection decision?

**What they're really testing:** whether quality in isolation is a criterion you would accept.

**Model answer:**
- **A quality number with no cost or latency attached is not a selection criterion**, because you
  cannot deploy it. This is why the composite boards that carry cost, latency, speed and context
  alongside quality are the most useful of the four leaderboard types.
- The three numbers that must be read together: **quality, price per million tokens (input and output
  separately, since they differ), and latency.** For a streaming product, **time to first token** is
  the latency the user experiences; for a batch product, total latency and throughput are the ones
  that matter.
- **Read the inference budget on the board itself.** A model evaluated with an unlimited reasoning
  budget and one evaluated with a capped budget are not comparable, and the board may mention this
  only in a footnote.
- **Watch the cost multipliers that hide in the configuration:** pass@k and majority@k multiply
  inference cost by *k*; chain-of-thought multiplies output tokens substantially; longer context
  multiplies input cost on every call. A score earned with majority@k at k = 8 is a system that costs
  eight times as much per query as the pass@1 number beside it on the same board.
- **Do the arithmetic on your own traffic, not the benchmark's.** Benchmark cost is measured on
  benchmark prompts, which are short. Your production prompts are not.
- **CS-12 gives the full worked arithmetic, and it is the one to reproduce in an interview.** A
  Text-to-SQL model priced at **$10 per million input / $50 per million output**, with an estimated
  **400 input and 100 output tokens per query**:
```
per query   = (400/1e6 × $10) + (100/1e6 × $50)
            = $0.004            + $0.005            = $0.009
per day     = $0.009 × 50,000                          = $450
per month   = $450 × 30                                = $13,500
in rupees   = $13,500 × 95                             ≈ ₹12.82 lakh/month
budget      = ₹3 lakh/month   ->  ~4x over  ->  you need ~1/4 the price
```
  That single computation is the whole argument: a model that **looks** affordable per token is **4×
  over budget** per month, and the conversion from a per-token rate to a monthly number is what makes
  it visible. Do it before the eval, not after.
- **Prompt caching is the first lever, and the arithmetic is stark.** Caching the static prefix — the
  schema, the system prompt — turned a **$10** input rate into **$1** on cache hits, with a
  **5-minute TTL**. For a Text-to-SQL workload where the schema is identical on every call, most of the
  input is cacheable, so this is where the 4× gap actually gets closed. Name it in that order: **cost
  model → shortlist → caching → custom eval**, because caching changes which models clear the filter.
- **Note the caveat that makes this a teaching artifact rather than a study.** CS-12's own numbers are
  internally inconsistent in several places — the daily volume is stated as both **50,000** and
  **5,000**, the cache-write price as both **$50** and **$12.50**, and three different budgets (₹30
  lakh / ₹3 lakh / ₹1 lakh) appear without being reconciled. Say this out loud if you use the example:
  **the method is the transferable part, the figures are illustrative.** That is itself the lesson of
  the pack — treat any number, including a lecture's, as evidence to be checked.
- For any RAG system specifically, the retrieval side is usually the cheaper lever and the embedding
  model is the piece you should be reading **MTEB** for — the generative model's leaderboard position
  may be the least important number in the decision.

**Red flags:** choosing on quality alone; no TTFT for a streaming product; ignoring the k multiplier;
comparing per-token prices without the actual token counts.

### Q20. Static versus dynamic benchmarks — when do you need which?

**What they're really testing:** whether you know the structural defence against saturation.

**Model answer:**
- **A static benchmark's dataset is the same today as it was in the paper** — five years ago, in the
  lecture's phrasing. It is comparable across time, reproducible, and doomed to saturate.
- **A dynamic benchmark's dataset is updated** against the latest data or a defined time window. It
  resists saturation by construction, because the questions keep being replaced.
- **The trade is direct and unavoidable:**
  - Static buys **comparability** — you can compare this year's model to last year's on identical
    items. It pays in saturation and contamination.
  - Dynamic buys **freshness and discrimination** — the benchmark stays hard. It pays in
    comparability, because a score today and a score six months ago are on different datasets.
- **You need both, for different purposes:**
  - **Static** for tracking progress over time, for regression suites, and for any claim of the form
    "model A beats model B" — you need identical items.
  - **Dynamic** for tracking the frontier, for detecting genuine capability change, and for any
    benchmark you intend to keep quoting for years.
- **The private test set is a third option**, and it is the one HLE uses: the dataset is static, but
  a held-out portion is never published, so contamination cannot reach it. It buys resistance without
  giving up comparability — at the cost of the community's ability to verify the split.
- And a fourth, cheaper defence: **canary strings**, which do not prevent contamination but make it
  detectable after the fact.

**Red flags:** treating one as strictly better; no comparability cost from dynamism; no awareness of
private test sets or canaries.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. Design a benchmark for a capability that has no benchmark yet.

**What they're really testing:** whether you can construct, not just consume.

**Model answer:**
Start with the three rules, because they determine whether the benchmark is worth building:

1. **Correlate with real-world usefulness.** Avoid "IQ-testy" tasks; the test is whether the task
   mimics real work. A benchmark that correlates with nothing useful will be gamed and then ignored.
2. **Launch as hard as possible.** Launching at 40% accuracy means you are targeting a capability the
   big labs already know about — start near **0–1%**.
3. **Make the answer deterministically verifiable.** Not only a cost argument: judges prefer their own
   outputs and are neither accurate nor robust. "Don't be lazy, just think about it more until you
   figure out how to have a deterministic verifier."

Then the construction, which is the four components of Q1 made concrete:

- **Dataset + task**: what is the item, what is the answer, and what is the model asked to do? Hold
  back a **private split** from day one, and consider **canary strings**.
- **Run configuration**: fix and document zero/few-shot, CoT, temperature, max tokens, tool access,
  and the inference budget. The budget is part of the measurement, not metadata.
- **Scoring**: aim for **programmatic** verification; if a judge is unavoidable, use a binary
  criterion that domain experts would agree on, aggregate many criteria, validate the judge, and
  version it.
- **Aggregation**: decide up front whether the headline hides a weak category, and publish
  per-category scores.

Then the two things that determine whether it survives:

- **Calibrate the difficulty.** The window is: not 75% in two months (too easy), not stuck at 0% for
  five years (too hard, or miswired). Verify the wiring — does the event actually fire, is the trigger
  reachable, is the expected action annotated in the right order — **before** publishing a surprising
  zero.
- **Plan for its death.** A benchmark is a **perishable asset** — one to two years, sometimes two
  months. Write down now what it will be replaced by, and accept that benchmark construction is a
  continuous practice rather than a project. The five-stage difficulty curve is the map: school exams
  → college exams → human evals → days-of-human-work → **verifiable tasks nobody has ever performed**.

**Red flags:** building a benchmark with no held-out split; a launch difficulty chosen for good
publicity; a judge with no validation; no plan for saturation.

### Q22. A benchmark is saturated. What now?

**What they're really testing:** whether you know the five things you can actually do.

**Model answer:**
Saturation is detected by the cluster — every frontier model in a narrow high band, e.g. 95/94/92 —
plus a check of the invalid-question rate. There are five responses, in rough order of preference:

1. **Repair it.** This is exactly what MMLU-Pro did to MMLU: 4 options → **10** (reducing guess
   probability), rote questions → **reasoning** questions, 57 subjects → **14 balanced categories**,
   and ~12,000 questions. The repaired benchmark then shows a **20-point gap between reasoning and
   non-reasoning models** — it discriminates again. Repair is the cheapest option because the
   community already knows the capability.
2. **Replace it with a harder sibling** — GPQA replaced nothing but extended the frontier of depth;
   HLE combined breadth and depth and added a private test set.
3. **Raise the configuration** — chain-of-thought, more shots, tools. This is the weakest option
   because it changes the measurement rather than the difficulty, and it is adjacent to configuration
   gaming.
4. **Make it dynamic**, so the dataset refreshes and the benchmark cannot be memorised.
5. **Retire it, explicitly and on the record.** Continuing to quote a saturated benchmark because it
   is familiar is a named top-10 mistake.

The judgement to demonstrate is that **saturation is not a failure of the benchmark** — it is the
expected end of a successful one. A benchmark that never saturates was probably measuring something
that does not matter.

**Red flags:** continuing to quote the score; raising the difficulty by making the scoring stricter
(a different thing); no replacement plan; treating saturation as someone else's problem.

### Q23. How do you defend a benchmark against contamination?

**What they're really testing:** whether you know the layered answer rather than the single one.

**Model answer:**
Four layers, and none of them is sufficient alone:

1. **Private held-out split.** The structural defence. HLE holds back a test set; the score you
   report comes from the part nobody could have trained on. The cost is verifiability — the community
   cannot audit the split — which is why it is combined with layer 2.
2. **Canary strings.** A marker injected into the published dataset. It does not prevent
   contamination; it **detects** it, because the string surfacing in a model's output reveals the
   dataset was consumed in training. Be honest about the limit: it is **mostly not possible** to
   predict in advance whether a dataset entered a training process.
3. **Dynamic refresh.** Keep replacing items, so memorisation has a shelf life. This trades
   comparability for resistance (Q20).
4. **Detection at scoring time.** Look for the shape of memorisation: uniform performance across the
   easy and hard tails, correct answers to corrupted questions, or a discontinuity when a dataset
   version changes.

Two things to add that most candidates miss:
- **Ask which training stage leaked.** Pretraining filters do not catch **alignment-stage**
  contamination — a reliability benchmark can be contaminated by alignment data while the pretraining
  pipeline was clean.
- **Assume the public dataset is contaminated and design around it.** The condition is a conjunction
  (public dataset **and** a crawl containing it), but the safe working assumption for any dataset more
  than a few months old is that the crawl happened.

**Red flags:** "we filter the training data"; no held-out split; canaries described as prevention;
ignoring the alignment stage.

### Q24. Build an internal benchmark programme for an organisation. Where do you start?

**What they're really testing:** whether you can move from reading benchmarks to running one.

**Model answer:**
1. **Start from the product's own traffic, not from a public benchmark.** A public benchmark measures
   a model; your programme measures your system. The golden dataset comes from real requests,
   real failures and real edge cases — and a human always creates it, as a separate activity from
   executing the pipeline.
2. **Define your own constraints first** — the five-step procedure's step 1: application type,
   latency need, cost ceiling, context need, deployment constraint. The benchmark you build must be
   scored against these.
3. **Build the four components deliberately**: dataset + task, fixed run configuration, a scoring
   pipeline with an explicit **extraction** step, and an aggregation that publishes per-category
   scores rather than a single average.
4. **Choose the grading method per criterion, in cost order**: programmatic → judge (validated,
   binary, aggregated) → human. Human time is for red teaming, annotation and grey-area adjudication,
   not for bulk grading.
5. **Use an eval harness rather than hand-rolled plumbing** — `lm-evaluation-harness`, Inspect or
   HELM — so your results are standardised and comparable to the wider field where overlap exists.
6. **Smoke-test before you spend.** A `--limit 20` run costs a few rupees; a full 8,000-question run
   cost **₹2,300** in the lecture's own demonstration. Test the plumbing on 20 items before paying for
   the full set — and at scale, tier the cadence: free checks per commit, sampled judges on PRs, the
   full suite nightly.
7. **Wire it into the deployment decision.** A benchmark that does not gate a release is a dashboard.
   Define the regression threshold and the owner.
8. **Feed it from production continuously.** Harvest real failures into the golden set, or the
   internal benchmark drifts away from reality while looking green.
9. **Re-derive it when the harness, model or environment changes**, and re-validate the graders.
10. **Keep a held-out slice that nobody trains against**, and say so in the report.

**Red flags:** starting with MMLU; no extraction step; a benchmark that gates nothing; no production
harvest; hand-rolled runner with no comparability to the field.

### Q25. A model leads on GPQA-Diamond, which has 198 questions. How much should that move you?

**What they're really testing:** whether dataset size changes how you read a gap.

**Model answer:**
- **Not much, on its own.** GPQA's three sizes are **546 extended / 443 main / 198 diamond** — and
  diamond is the one quoted most often precisely because it is the hardest, which makes it the one
  with the smallest sample. On 198 questions, each item is worth about **0.5 percentage points**, so
  a three-point lead is six questions.
- **The rule is general: when the dataset is small, demand a confidence interval before believing a
  rank difference.** At n = 198, the standard error on a proportion near 0.5 is roughly 3.5 points —
  a gap smaller than that is noise.
- **It does not make the benchmark bad.** GPQA's value is real: it was built to be **Google-proof** —
  a non-specialist cannot solve a question in 30 minutes with search — and it captured a genuine
  capability jump (**GPT-4 at 39% → GPT-4o 56% → o1 78%**), which is exactly what a good benchmark
  does. The lesson is about the *reading*, not the benchmark.
- **A second reason for caution on any human-baseline claim:** GPQA's reported PhD baseline was
  **81.3%**, but when OpenAI hired PhDs to attempt the questions they averaged **69.7%**. A human
  baseline is a measurement with its own sample and its own selection effects, so "we beat PhDs" is a
  claim about a specific group of PhDs on a specific day.
- **The generalised answer:** a score gap should be read against three sizes — the **dataset size**
  (198 versus 14,000), the **interval** it implies, and the **number of runs** behind each number.
  Reporting a leaderboard rank without any of the three is rank bias.

**Red flags:** believing a 3-point lead on 198 questions; no confidence interval; treating a human
baseline as a fixed constant; reading rank without spread.

### Q26. What does "build your own methodology" actually mean in practice?

**What they're really testing:** whether you have an operational answer to the field's closing advice.

**Model answer:**
The instruction at the end of the source is explicit — **do not accept the number thrown in your
face; implement your own methodology and decide from that.** Made concrete, that is five things:

1. **A frozen dataset of your own traffic**, with your own requirements enumerated and your own
   golden answers.
2. **A fixed run configuration**, published alongside every result.
3. **A scoring pipeline with an explicit extraction step**, so a format change does not silently
   become a capability regression.
4. **An aggregation that reports per-category**, so a strong average cannot hide the category your
   users actually depend on.
5. **A baseline and a cadence** — because a number with nothing to compare against is an anecdote.

Two things to say out loud, because they are what separate a methodology from a ritual:
- **Your methodology's job is not to reproduce the leaderboard's ordering.** It is to predict *your*
  outcome. If your internal eval and the public board disagree, that is information about your use
  case, not an error to fix.
- **Your methodology will itself decay.** Goodhart applies to you too: the moment your internal
  benchmark gates a release, teams optimise for it. Contamination, saturation and gaming are not
  problems that happen to other people's benchmarks.

**Red flags:** defining methodology as "run MMLU ourselves"; no frozen dataset; no extraction step;
no cadence; assuming internal benchmarks are immune to Goodhart.

### Q27. Which benchmark should you report when the decision is about a RAG system?

**What they're really testing:** whether you match the measurement to the component.

**Model answer:**
The RAG system has at least three separately-evaluable components, and each has a different board or
measurement:

| Component | What to read | Why |
|---|---|---|
| **Embedding / retriever** | **MTEB** | The embedding-model ranking leaderboard, and the recommended board for RAG work |
| **Generator** | A general capability board, ideally one carrying cost and latency | Quality is only one of three selection criteria |
| **Tool / function calling**, if the system calls tools | **BFCL** (Berkeley Function Calling Leaderboard) | The tool-calling capability ranking |
| **The pipeline as a whole** | Your own evaluation | Recall@K, faithfulness, answer relevance are component metrics no public board reports for your corpus |

The candidate-level point: **the leaderboard position of the generative model is often the least
important number in a RAG decision.** Retrieval quality caps answer quality — a perfect generator with
bad retrieval produces confident nonsense — and the embedding model is usually cheaper to change. A
strong answer names MTEB for the retriever before naming any generative benchmark.

And the closing discipline: benchmark the components, but **decide on the pipeline.** A model that
wins every component board can still lose on the assembled system, because the assembly is where the
harness confound lives.

**Red flags:** quoting only a generative-model board for a RAG decision; ignoring retrieval; no
end-to-end eval; treating BFCL or MTEB as a general capability score.

### Q28. What is the open problem in benchmarking?

**What they're really testing:** whether you know where the frontier actually is.

**Model answer:**
Four, and a good answer separates the structural from the temporary:

1. **Perishability.** Every benchmark saturates — one to two years, sometimes two months — and there
   is no known way to build one that does not. The best available answers are **dynamic datasets**,
   **private test sets** and **continuous construction**, and none of them removes the problem;
   together they only manage it.
2. **Contamination is undetectable in general.** You cannot reliably determine whether a dataset
   entered a model's training process, and closed models give you no way to check. Canary strings
   detect a specific failure mode after the fact; they do not establish cleanliness.
3. **The difficulty ceiling.** The five-stage curve — school exams → college exams → human evals →
   days-of-human-work → **verifiable tasks nobody has ever performed** — has a stage 5, and **nobody
   has yet said what stage 6 is.** Building benchmarks for tasks with no known solution is a genuine
   open problem.
4. **The evaluation of open-ended work.** Deterministic verification is the goal and it is only
   available for a subset of tasks. For everything else, the field is left with judges, and judges
   are neither accurate nor robust enough to be trusted for capability measurement — which is why the
   strongest current position is "make the answer deterministically verifiable, and don't be lazy
   about finding out how."

Two structural consequences worth naming: **the human baseline is not a fixed constant** (GPQA's
PhD baseline measured 81.3% in the paper and 69.7% with a different group of PhDs), and **the field
has no shared disclosure standard**, which is why scores for the same model do not match across labs.

**Red flags:** naming only "benchmarks get gamed"; no distinction between structural and temporary
problems; claiming deterministic verification is always achievable; no awareness of the stage-6 gap.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Two labs report different scores for the same model on the same benchmark. Diagnose.

**What they're really testing:** whether you can enumerate the causes in order of likelihood.

**Model answer:**
The candidate causes, ordered by how often they turn out to be the answer:

1. **Different harness.** The scaffold moves scores more than most people expect, and can invert
   rankings between two models. This is the most common cause and the most commonly omitted from
   reporting.
2. **Different run configuration** — shots, CoT, temperature, max tokens, tool access. A 5–10% swing
   is available from configuration alone, and a lab's own model frequently gets the favourable
   settings.
3. **Different scoring path** — generation versus log-probability produced **84% versus 87% on MMLU
   for the same model**. This alone explains most same-benchmark disagreements.
4. **Different scoring strategy** — pass@1 versus pass@k versus majority@k.
5. **Different dataset version** — regenerated subsets, repaired items, dropped questions. Check the
   fine print: a documented case is a reported score with an admission that **40 of 237** problems were
   omitted.
6. **Different extraction or grading method**, including a judge at a different version. Judge-graded
   results should not be compared across years at all.
7. **Different quantization or an unannounced provider-side model update.**
8. **Different provider** — a score produced through an inference API measures the provider, and
   will not reproduce locally.

The fix at the ecosystem level is a **shared disclosure schema**: provenance, model version,
quantization, evaluation library, configuration, scoring path, grader version, per-instance results.
At your own team's level it is the same discipline applied to your own benchmark. And the meta-lesson:
**read the fine print** — eval nuances hide in footnotes and in charts not scaled to reality.

**Red flags:** assuming one lab is lying; no confidence interval check; no configuration comparison;
trusting a chart without error bars.

### Q30. A model scores 95% on a knowledge benchmark but fails constantly in your product. Diagnose.

**What they're really testing:** whether you can separate four distinct explanations.

**Model answer:**
Four explanations, and they are not alternatives — usually more than one applies:

1. **You measured a model and deployed a system.** The harness, prompts, retrieval and tooling are
   yours. The benchmark held them constant or absent; your product does not.
2. **Saturation.** Check the cluster: if the top models are all between 92 and 94, the benchmark
   stopped discriminating, and the 95% is a membership card rather than a measurement. Also check the
   invalid-question rate — for MMLU, **6.5–8% of questions are wrong**, which caps everyone near 92.
3. **Contamination.** If the dataset is public and old, the score may be partly memorisation. The
   tell is usually that the model is as good on the hard tail as the easy tail, or that it handles
   corrupted questions correctly.
4. **The wrong capability.** A knowledge benchmark does not measure ambiguity handling, long-horizon
   behaviour, tool use, or the ability to ask a clarifying question — which is usually what production
   failure looks like. The Kaggle analogy applies exactly: clean data, clear problem statement, easy
   work — versus messy production.

The remedy is the same in all four cases: **stop treating the public score as a predictor and build
your own evaluation on your own traffic**, with your own requirements enumerated and graded. And
before you do, use the public score for the one thing it is good at: **shortlisting 3–5 candidates.**

**Red flags:** blaming the model; no saturation or contamination check; assuming a knowledge score
predicts application behaviour; no internal eval.

### Q31. Your internal benchmark went from 60% to 92% in a month, with no model change. Diagnose.

**What they're really testing:** whether you suspect the instrument before the system.

**Model answer:**
A jump that large with no model change is a **measurement** change, and the causes are enumerable:

1. **Contamination of your own benchmark.** Your internal dataset leaked into a fine-tuning run, a
   prompt, a few-shot example set, or a retrieval index that the system now consults. **Any environment
   or dataset used as a training signal stops being a valid measurement** — this is the single most
   likely cause of a suspicious internal jump.
2. **An extraction regression in the other direction** — the extractor became more permissive, or the
   output format changed to one the extractor matches more easily. Check the extractor version, not
   only the model.
3. **Configuration drift** — a shot count raised, CoT enabled, tools switched on, temperature moved to
   0, a max-token cap lifted.
4. **The grader changed**: a judge upgraded, a rubric relaxed, a human annotator replaced. Judge drift
   is a real cause of upward movement.
5. **The dataset changed** — items dropped, a subset regenerated, the hard category quietly removed.
6. **Reward hacking**, if the system is an agent that can see any part of the grading path or the
   files it is graded on. Check for test-file modification (fingerprints) and for a pass rate rising
   while task quality does not.
7. **The baseline moved** — a caching layer, or your scoring pipeline now returning a default value for
   timeouts, which reads as success.

The right response: **freeze the model, re-run the old configuration on the old dataset, and see if
the old number reproduces.** If it does not, the instrument moved. Then read the per-item
differences rather than the aggregate — a 32-point move is never uniform.

**Red flags:** celebrating the jump; no extractor versioning; no fingerprinting; not checking whether
the dataset or grader changed.

### Q32. A vendor claims their model "beat humans on JEE/NEET and GPQA." Evaluate the claim.

**What they're really testing:** whether you can dismantle a marketing claim without being cynical
about the underlying result.

**Model answer:**
Four things to check, and the claim usually survives some of them:

1. **What does the exam actually test?** AGIEval's whole design was to measure LLMs on existing human
   exams, and the value is comparing to a **measured** human baseline — an **average of 67%** and a
   **top baseline of 91%** across 20 exams. "Beat humans" usually means "beat the average," not the
   top. Ask which.
2. **Is the human baseline a constant?** It is not. GPQA's paper reported a PhD baseline of **81.3%**;
   when OpenAI hired PhDs to attempt the same questions they averaged **69.7%**. A baseline is a
   measurement with its own sample and selection effects.
3. **What are the run conditions, and are they symmetrical?** Configuration gaming gives a 5–10%
   swing; a lab comparing its own best configuration to a human's untimed/untooled attempt is not a
   comparison.
4. **What else does the exam not test?** An exam is a closed-world instrument. Passing JEE/NEET says
   nothing about ambiguity handling, long horizons, tool use, or production messiness — the same
   limits as any benchmark.

**Is the result still meaningful?** Often yes, and a strong candidate says so: **GPT-4 went from 39%
to o1's 78% on GPQA**, which is a genuine capability jump on questions designed to be **Google-proof**
— unsolvable by a non-specialist with search and 30 minutes. The correct posture is not to dismiss the
result but to **narrow the claim to what was measured**: a specific model, on a specific set of
questions, against a specific comparison group, under specific conditions.

**Red flags:** accepting "beat humans" as a capability claim; not asking which human baseline;
ignoring symmetry of conditions; dismissing the result entirely rather than narrowing it.

### Q33. A full benchmark run cost ₹50,000 and took three days. Diagnose and fix.

**What they're really testing:** whether you treat evaluation cost as an engineering problem.

**Model answer:**
The lecture's own numbers give the calibration: a **20-item `--limit` smoke test costs ₹3–4**; a full
**8,000-question GSM8K run costs ₹2,300**. So ₹50,000 is roughly 20× a full GSM8K run and signals
structural waste, not a big benchmark. Diagnose in this order:

1. **Was there a smoke test?** A `--limit 20` run verifies the plumbing for a few rupees. Running the
   full set first is the named mistake. **Always smoke-test before paying.**
2. **Retry policy.** Retrying permanent HTTP failures (`400, 401, 403, 404, 422`) burns budget on calls
   that can never succeed. Cap retries and make failures fail fast.
3. **Batch and rate-limit handling.** The loop "is not as simple as it looks" — extraction, batching,
   retries and rate limits all cost. Serial unbatched calls to the same provider is the usual culprit.
4. **Was a judge used where a programmatic check would do?** Judge calls are an order of magnitude
   more expensive and slower, and are the most common reason an eval bill is inexplicable.
5. **Is the eval being run in CI?** Long or expensive runs belong on a nightly or pre-release cadence
   with a tiered structure: free checks per commit, sampled judges on pull requests, the full suite
   nightly.
6. **Is the run repeated unnecessarily?** Three runs with ± is the reporting standard, not thirty.

The fixes: **use an eval harness** rather than hand-rolled plumbing so batching and retries are
already handled; **smoke-test with `--limit`**; **reserve judges for the language-shaped residue**;
and **size the run to the decision** — a regression gate does not need the full public set.

**Red flags:** no smoke test; no retry policy; judges everywhere; the full suite in CI; unbounded runs.

---

## Tier 5 — Trap questions (5 Q)

### Q34. *(Trap)* "Our model tops the leaderboard, so we should ship it."

**The naive answer:** agree — the top model is the best model.

**What they're testing:** whether you know a leaderboard is a filter, not a decision.

**Model answer:**
- The lecture's own warning, aimed exactly at this reflex: "most of the time, when you go to build an
  application, step one is model selection — and in model selection, step one is going to the
  leaderboards to see which model is on top. So we develop a bias inside us that if it is on the
  leaderboard, it must be right. But that is not the case."
- **Leaderboards are a filtering tool, not a selection tool.** The correct use is: shortlist 3–5, then
  run your own eval.
- Why the top row is not automatically right:
  - **Small gaps are noise.** 84.3 at rank 3 versus 84.1 at rank 5 is a 0.2-point difference and means
    nothing without a confidence interval.
  - **The benchmark may be saturated** — everyone in the 92–94 band means the ranking is a coin toss.
  - **It may be contaminated** — a public dataset plus an internet-scale crawl is a memorisation
    opportunity.
  - **The configuration is unknown** — 84% or 87% on the same benchmark depending on the scoring path.
  - **Composites hide their weights**, so "top of the board" may mean "top of the weights someone
    chose."
  - **It measures a model, and you are deploying a system.** The harness and pipeline are yours.
- Then the practical answer: **check cost and latency**, because the top model is usually the most
  expensive per token, and the second-best on a benchmark you actually care about is frequently
  sufficient at a fraction of the price.

**Red flag:** treating the leaderboard position as the decision.

### Q35. *(Trap)* "We scored 92% on MMLU, so it's a strong reasoner."

**The naive answer:** agree — high scores mean high capability.

**What they're really testing:** whether you know what a knowledge benchmark measures — and does not.

**Model answer:**
- **MMLU measures knowledge breadth, not reasoning.** It is the parametric-knowledge test: 57 subjects
  at a basic level. The capability it opened was *breadth*, and depth needed GPQA.
- The 92 is also not what it appears:
  - **The benchmark is saturated at that level.** Frontier models sit in the 86–92 band, and the top 10
    cluster between 92 and 94. A 92 is a membership card, not a discrimination.
  - **~6.5–8% of MMLU's questions are simply wrong**, capping every model near 92% regardless of
    capability. Treating that as a ceiling on the model is one of the named top-10 mistakes.
  - **The scoring path alone accounts for 3 points** — 84% by generation versus 87% by log-probability
    on the same model.
- **The right counter-example is SimpleQA**: the same model that scores **88% on MMLU scores 40%** when
  asked short-answer factual questions with no options. If 92% were a general capability claim, the
  40% would be impossible.
- And the MMLU-Pro repair is instructive: moving from 4 options to 10 and from rote questions to
  reasoning questions opened a **20-point gap between reasoning and non-reasoning models.** The
  original MMLU could not see that gap — which is precisely the evidence that it was not measuring
  reasoning.

**Red flag:** reading a knowledge score as a reasoning or agentic score.

### Q36. *(Trap)* "We'll use a private benchmark, so it can't be contaminated or saturated."

**The naive answer:** agree — private data is safe data.

**What they're testing:** whether you know the two failures have different remedies.

**Model answer:**
- The two problems are separate, and privacy only helps with one:
  - **Contamination** — a private dataset genuinely reduces the risk, because it was not in any public
    crawl. This part of the claim is fair.
  - **Saturation** — **privacy does nothing.** Saturation happens when models get better, not when
    questions leak. A private benchmark saturates on exactly the same schedule as a public one, and you
    have no community watching to tell you it happened. **Every benchmark is perishable**, and a
    private one is perishable without an audience.
- Three further costs of privacy:
  - **No verifiability.** With a private dataset, nobody can check what happened. If someone games a
    benchmark in a reproducible way, a public dataset lets the community detect it; a private one does
    not.
  - **The evidence that public benchmarks saturate faster than private ones is weak.** One study found
    no strong correlation between public/private status and saturation speed, so you may be paying a
    cost for a benefit that does not exist.
  - **No comparability to the field.** A private score cannot be checked against anything, so it cannot
    be a sanity check on your own pipeline.
- **The nuance that resolves it: the reason to publish is verifiability, not saturation resistance.**
  The practical middle is to publish the **methodology, tooling and rubrics** even when the seed data
  cannot be published, and to hold out a **private slice** for your own reporting. That gets both —
  public verifiability of the method and a clean measurement of your system.

**Red flag:** believing privacy solves saturation.

### Q37. *(Trap)* "Just average the subject scores — it's the same thing."

**The naive answer:** agree — an average is an average.

**What they're testing:** whether you have ever computed a weighted benchmark score.

**Model answer:**
- It is not the same thing. MMLU has **57 subjects**, and they do not contain equal numbers of
  questions. The correct computation weights by question count:
```
overall = SUM(n_s * score_s) / SUM(n_s)
```
  An unweighted mean over subject percentages over-weights every small subject, so a model that
  happens to be strong on the three-question subjects gets a free boost.
- **The larger problem is what any average hides.** A 92% average across 57 subjects is compatible
  with 99% everywhere except one subject at 40%. If that subject is the one your product depends on,
  the headline is actively misleading. This is **aggregation hiding** — one of the four named reasons
  a benchmark number is untrustworthy — and the remedy is to publish **per-subject scores**, not only
  the average.
- **It also breaks comparability.** Two labs averaging differently is one of the many reasons the same
  model reports different numbers in different places.
- The generalised rule: **an aggregate is a claim about a weighting**, whether or not a weighting was
  intended. When you see a composite score, ask what the weights are — and if the board does not say,
  that is itself the answer: the weighting is undisclosed, so the score is an opinion.

**Red flag:** averaging percentages without weights; quoting an average with no per-category data.

### Q38. *(Trap)* "We used majority@k for our model and pass@1 for theirs — both are valid, so it's a fair comparison."

**The naive answer:** agree — both metrics are legitimate.

**What they're testing:** whether "valid individually" and "comparable to each other" are the same
claim. They are not.

**Model answer:**
- **Both metrics are legitimate. The comparison is not.** The governing rule is absolute: "when you run
  both models, **all the settings must be exactly the same**." A benchmark number is only comparable
  to another number taken the same way.
- How large is the distortion? **majority@k at k = 8 asks the same question eight times** and takes the
  mode. For any model with per-trial accuracy above chance, that is worth several points — and the gap
  between adjacent leaderboard ranks is typically **0.2 to 2 points**. The measurement difference
  dwarfs the thing being measured.
- **There is also a cost consequence the framing hides.** majority@k at k = 8 is a system that costs
  roughly **eight times** as much per query. So the comparison is not merely unfair on quality — it
  compares a cheap system to an expensive one and reports only the quality axis.
- **And a deployment consequence.** pass@k and majority@k are only legitimate deployment strategies if
  you actually deploy them. If your product makes one call per query, a majority@k score describes a
  system you do not run. Reporting it as your model's capability, against a rival's pass@1, is
  configuration gaming — a **5–10% swing** available from configuration alone, and one of the four
  named reasons a benchmark number is untrustworthy.
- The correct answer in the room: **ask which scoring strategy was used for each model.** If the answer
  differs, the comparison is not available, and you say so rather than reporting a ranking you cannot
  defend.

**Red flag:** accepting "both are valid metrics" as a defence of an asymmetric comparison.

---

## Live-coding / whiteboard prompts (3)

### Prompt 1 — Read a leaderboard row and produce a shortlist (45 min)

> Here is a row from a multi-benchmark leaderboard: a model name, a composite score of 71.2, a price
> per million tokens, a median latency, and a context length. Your team is building a customer-support
> RAG system with a 2-second P95 latency budget, a hard per-query cost ceiling, and an on-premise
> deployment requirement. Produce a shortlist and justify it.

**What a strong answer covers:** state the **five-step procedure** and work it in order — constraints
first (latency budget, cost ceiling, on-premise constraint, context need), then board choice, then
definitions, then shortlist, then your own eval. Name what the row is **missing**: the **run
configuration** (shots, CoT, temperature, tools), the **scoring strategy** (pass@1 or pass@k), the
**scoring path** (generation vs log-probability — worth 3 points on MMLU alone), the **composite
weights**, the **confidence interval**, the **dataset age**, who ran it, and whether it is **saturated**
(check the spread of frontier models — a 92–94 cluster means the ranking is noise). Then apply the
constraints: the **on-premise** requirement may eliminate the leader outright; the **cost ceiling**
eliminates the most expensive models; the **latency budget** eliminates anything whose median plus a
generation is above 2s. Note that for a RAG system you should also be reading **MTEB** for the
embedding model, and that retrieval quality caps answer quality. Close by saying the shortlist is
**3–5 models to be evaluated on your own traffic** — the leaderboard filtered, it did not decide.

**Red flags:** picking rank 1; ignoring the on-premise constraint; no confidence interval; no mention
of the missing configuration; treating the composite score as transparent.

### Prompt 2 — Design a custom eval for your own use case (60 min)

> Your team ships an internal code-review assistant. Design the evaluation you will run to choose a
> model, and the one you will keep running after launch. Say what you measure, how you score it, and
> what would make you distrust your own number.

**What a strong answer covers:** start from **your own traffic**, not a public benchmark, and say why
(the Kaggle analogy — clean data and clear problem statements do not predict messy production). Build
the **four components** deliberately: dataset + task (real review requests with gold outcomes,
requirements enumerated), **fixed run configuration** (shots, temperature ≈ 0, max tokens above CoT
truncation, tools), **scoring with an explicit extraction step**, and **aggregation reporting
per-category** rather than one average. Choose the **grading method per criterion in cost order** —
programmatic where decidable (does the suggested fix compile, does it pass the tests), a **validated
binary judge** only for the language-shaped residue, human only for red teaming and grey areas. Use an
**eval harness** rather than hand-rolled plumbing; **smoke-test with `--limit`** before paying for the
full run. Report **score ± spread over N runs with the harness named**, and compare against a
**baseline**. Then answer the distrust question properly: **your own number can be contaminated**
(if it leaked into a fine-tune or a few-shot prompt), **saturated** (if scores converge), **gamed**
(if it gates a release, teams will optimise it — Goodhart applies to you), and **broken by an
extraction regression** (a format change that reads as a capability drop). State the defence for each:
a held-out slice, a dynamic refresh, fingerprints and grader isolation, and extractor versioning.

**Red flags:** starting from MMLU; hand-rolled runner; no extraction step; a bare score with no spread
and no harness; no plan for the eval decaying.

### Prompt 3 — Audit a benchmark claim (30 min)

> A vendor's release note says: "Our new model achieves 94.1% on MMLU, 78% on GPQA-Diamond, and leads
> the leaderboard. It surpasses human experts."

**What a strong answer covers:** decompose the claim into its four components and check each. **MMLU
94.1%**: is the benchmark saturated at that level (frontier band is 86–92; top-10 cluster 92–94)? Is
the **invalid-question rate** (~6.5–8% of MMLU questions are wrong) being read as a capability
ceiling? **Which scoring path** — generation or log-probability, worth 84 vs 87 on the same model? How
many shots, was CoT on, what tools? **GPQA-Diamond 78%**: diamond has only **198 questions**, so a
3-point lead is about six questions and needs a confidence interval; and it is a genuine capability
signal (GPT-4 was 39%, o1 was 78%), so the number itself is plausible — the claim around it is what
needs narrowing. **"Leads the leaderboard"**: which board, which type, and are the **composite weights**
disclosed? Is the gap larger than the interval? **"Surpasses human experts"**: which experts — GPQA's
paper reported an **81.3%** PhD baseline, while OpenAI's hired PhDs averaged **69.7%**. A human
baseline is a measurement, not a constant, and "experts" is doing a lot of work in that sentence.
Close by **narrowing rather than dismissing**: a specific model, on specific questions, against a
specific comparison group, under specific conditions — and then say the decision still requires your
own eval, because the benchmark measured a model and you are deploying a system.

**Red flags:** accepting or dismissing the whole claim; no confidence interval on 198 questions; not
asking which human baseline; no configuration check.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** A vendor has supplied a support-ticket triage model with a glossy results page: **91.2 on a
> composite capability board**, **top-5 on that board**, **₹9 per million input tokens**, **38 tok/s**
> output, and the line *"outperforms human agents on our internal benchmark."* Your company wants to
> use it to route **12,000 tickets a day** across **9 product lines** and **6 languages**, with a
> **P95 of 2.5 seconds** and a **hard ceiling of ₹0.40 per ticket**. There is no internal evaluation
> today. The vendor will not share the evaluation harness.
>
> **Deliverables (≤6 pages):**
> 1. **The audit of the vendor's page.** Name every missing disclosure (configuration, scoring
>    strategy, scoring path, extraction, grader, aggregation, weights, interval, dataset age,
>    harness), say which of the **four untrustworthy reasons** could be inflating 91.2, and state what
>    the composite score does *not* tell you about triage.
> 2. **The capability map.** Which capabilities you actually need (multilingual handling, ambiguity
>    handling, escalation judgement, structured output) and which public benchmarks, if any, speak to
>    each — including the ones that do not exist.
> 3. **Your own evaluation.** The dataset (source, size, how the golden answers are made, the held-out
>    slice), the **four components**, the per-criterion grading method in cost order, and the
>    **extraction** step.
> 4. **The cost and latency model.** Do the arithmetic against ₹0.40 and 2.5s P95 at 12,000
>    tickets/day, including prompt length, output length and retries; say whether the vendor's numbers
>    are compatible with the ceiling, and what k-multiplier would break it.
> 5. **Your own number's failure modes.** How your internal benchmark could be contaminated,
>    saturated, gamed or silently broken — and the defence for each.
> 6. **The decision rule.** The threshold at which you would adopt, adopt-with-fallback, or reject —
>    and the one new piece of evidence that would change your answer.
>
> **What is being assessed:** whether you treat the score as evidence rather than fact; whether you can
> produce the disclosure list from memory; whether you separate **filtering** (the leaderboard) from
> **deciding** (your own eval); whether you read a **human baseline** as a measurement; and whether
> your arithmetic connects the benchmark to the deployment constraint rather than restating it.

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Mid hire | Senior hire |
|---|---|---|---|
| **Definitional precision** | "A test for LLMs" | Knows a benchmark has a dataset and a score | Names **four components** and refuses to read a number without all four |
| **Disclosure instinct** | Accepts the number | Asks which benchmark | Produces the full checklist — configuration, scoring path, strategy, extraction, grader, aggregation, interval, provenance, who ran it |
| **Saturation literacy** | A high score is good news | Knows benchmarks saturate | Detects the 92–94 cluster, checks the invalid-question rate, and says "perishable asset" with the replacement plan |
| **Contamination** | "Public is fine" | Knows contamination exists | Explains the public-plus-crawl conjunction, names canaries and private splits, and asks **which training stage** leaked |
| **Historical fluency** | Names MMLU and GPQA | Knows what each measures | Tells the causal chain — each benchmark built to fix the last one's failure, TruthfulQA's bigger-models-less-truthful finding, MMLU-Pro's 20-point reasoning gap |
| **Selection discipline** | Picks rank 1 | Shortlists a few | Runs the three-stage procedure in order, requirements first, applies a **budget filter** to the shortlist, and **filters before deciding** |
| **Custom model evals** | Takes the board's word | Knows you should test on your own data | Computes the per-query → monthly cost arithmetic, uses **prompt caching** to close a budget gap, and grades by **executing** the output rather than comparing strings |
| **Statistics** | One score | Knows about averaging | Weights by question count, demands per-category scores and confidence intervals, and knows a 0.2-point gap is noise |
| **Cost and latency** | Quality only | Knows price matters | Reads quality, price and latency together, knows the **k-multiplier**, and does the arithmetic on real traffic |
| **Construction** | Only consumes benchmarks | Knows a good benchmark is hard | States the three rules — real-world correlation, launch at 0–1%, deterministic verification — and can design one |
| **Meta-awareness** | Trusts benchmarks | Reads the fine print | Knows Goodhart applies to **their own** internal benchmark too, and that a benchmark that gates a release will be gamed |

**The single strongest signal in this domain:** the candidate treats a benchmark score as **evidence
rather than a fact** — they can say what four components produced it, what four forces could be
inflating it, and why the honest next step is to run their own evaluation on their own traffic, because
the benchmark measured a model and they are deploying a system.
