# Interview Bank: Reward Models, Verifiers & Best-of-N

> `T05` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T05-verifiers-best-of-n.md) · [Case study](../01-case-studies/T05-verifiers-best-of-n.md) · [Design blueprint](../03-design-blueprints/T05-verifiers-best-of-n/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer here is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a transcript statement with the source named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here with assumptions shown. Where the corpus does not supply a figure, the answer says so rather than inventing one.

Two ASR caveats travel with this topic: the lecture renders one paper's author as "**Barami at all**" (Beirami et al.) and lecture 11 renders a collaborating institution as "**Sweden**." Both are transcription artefacts; say the correction rather than the spelling.

Questions are ordered to read as one interview: foundations, then mechanism, then the tradeoffs, then debugging, then design at scale.

---

### Foundations

#### T05-Q1 · What is best-of-N, and what is the one thing it forbids?
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** Describe best-of-N in one breath. Then tell me the constraint on the ranking signal, and why that constraint defines the whole design.

**Model answer:** Sample `n` outputs from the generator, rank them by a score, return the top one — and the score must come from **something other than the model's own log-probability**. The lecture states the requirement flatly: "We're going to sample n things, we're going to rank them according to something other than the log probability from our original model because if we were doing that, we would just be doing some kind of variant of **mode-seeking search**" `[T]` (CMU lecture 12). That prohibition is the design, because it means every question in this topic reduces to *what does the judging, and can you trust it*. Using log-probability selects the most likely sample, not the best one — the same likelihood trap that keeps beam search out of production `[T]` (CMU lecture 4). Everything else is arithmetic: `n` is a cost parameter, the pool is a diversity problem, and the scorer is a model with its own failure modes. The second half of the framing is what makes the approach tractable: best-of-N is a special case of rejection sampling, and because we only care about **relative** behaviour, we never need an accurate probability — only a correct ordering. That is why a fixed `n` with a possibly-bad answer is the accepted contract.

**Signal:** Names "a score that is not the model's log-probability" as the defining constraint and immediately connects it to mode-seeking — rather than describing best-of-N as "generate several and pick the best one," which is what a weak answer does.
**Follow-ups:**
- *Why is log-probability mode-seeking rather than quality-seeking?* — It selects the most probable sample; the likelihood trap is Q1's mechanism — see [T01](./T01-sampling-decoding.md).
- *What licenses the fixed `n`?* — Rejection sampling with only relative behaviour required; Q2 carries the formalism.
**Red flags:** Says best-of-N is "pick the highest-scoring sample" with no statement about what may not do the scoring; treats `n` as a quality dial.

#### T05-Q2 · Best-of-N as rejection sampling — and where the formalism breaks
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Frame best-of-N as rejection sampling: what are the proposal, the target, and the acceptance rule? Then tell me where that framing stops being valid.

**Model answer:** The target `D` is the distribution of human preference — a distribution "kind of hard in practice to write down" and hard to sample from directly `[T]`. The proposal `P` is the generator, and its support must be "at least as large as D's support," so that "everywhere that D has nonzero probability, we need to have non-zero probability" `[T]`. A sample is accepted with probability `D(x) / (C · P(x))`, where `C` is "the tightest upper bound" large enough that target probability never exceeds `C` times proposal probability, so the ratio is a well-defined probability in `[0,1]` `[T]`. The lecture explicitly contrasts this with constrained decoding, where rejection sampling was the *wrong* tool because "you could wind up kind of stuck in this space where you're generating forever" `[T]` — here it is fine because we want only relative behaviour and `n` is fixed, so we "just return the sort of least bad of our samples" `[T]`. **Where it breaks:** the single-proposal assumption. Sampling at several temperatures or from several models means "you no longer have sort of one proposal distribution" — the lecture's example is a "best of 50" drawn 20 from one model, 10 from another, 20 from the first model under different decoding settings `[T]`. Formally the framing is void; practically "this is still quite effective in practice" `[T]`. The engineering response is to record the mixture in the ledger rather than hide it.

**Signal:** Separates the relative-behaviour licence (which is what makes the bounded budget acceptable) from the single-proposal assumption (which is what the mixture breaks), and knows the formalism is broken-but-workable rather than fatal.
**Follow-ups:**
- *Why was rejection sampling rejected for constrained decoding but accepted here?* — The generate-forever risk versus a fixed `n` and a relative-only requirement.
- *What do you do about the break?* — Vary temperature where *diversity* is the purpose, keep the proposal fixed where *auditability* is; log the mixture — Q13.
**Red flags:** Cannot name the acceptance ratio or `C`; presents multi-temperature sampling as costless; claims rejection sampling is never appropriate.

#### T05-Q3 · Bradley-Terry, and the two failure modes that ship bugs
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** How is a scalar reward model built, what is the training recipe, and name the two ways it is ill-defined in practice.

**Model answer:** The construction is Bradley-Terry: "the probability that thing I is better than thing J is related to this ratio of probabilities between them," with the sanity check that equal probabilities give "a 50-50 shot" `[T]`. Restated in log space — the probability that `y1` beats `y2` is "exactly the same equation if your rewards are log probabilities" `[T]` — which is where "reward = log-probability" comes from and why reward models are "often just classification heads." The recipe: "take an existing model, generally a relatively strong language model. If you want to get a scalar prediction you'll strip off the language modeling head and replace it with some kind of sequence classifier," then train on pairwise preference data — "one input and two outputs, one that is chosen and one that is rejected" `[T]`. **The two failure modes.** First, reward models are "not always well defined over **subsequences**" — the reward curve over increasing prefixes is "really really bouncy. It goes negative at several points" `[T]`. Never score a partial draft. Second, **there is not always a meaningful pairwise distinction**: "if you ask a model to name the color of the rainbow, is green or blue a better answer… by and large these are about the same quality output," and the expected scalar is "somewhere around 0.5" `[T]`. Build the tie band into the ranker on day one; shipping a `>` comparison and discovering ties later is the classic bug. Scope caveat worth voicing: this is "one variety of reward model… a reward model can be any model that predicts a reward" `[T]`, and the scalar variety is "a little less common than it was a year or two ago" `[T]`.

**Signal:** Gives the log-space identity (not just the ratio) and volunteers both failure modes — the bouncy subsequence curve and the ~0.5 tie case — as design constraints rather than trivia.
**Follow-ups:**
- *Why does the log-space identity matter operationally?* — It is why a classifier head is a valid reward model, which is what makes the trained option cheap — Q15.
- *What does ~0.5 mean for the ranker?* — A tie band, and RewardBench v2's notion of ties — Q7.
**Red flags:** Describes Bradley-Terry as "a model that outputs a score" with no preference ratio; has never heard of the subsequence problem.

#### T05-Q4 · Generative reward models: the three cons, and which one actually bites
**Difficulty:** L3 · **Depth expected:** 5 min
**Question:** An off-the-shelf LLM-as-judge is the fastest way to get a ranker. Give me its pros, its three named cons, and then tell me which one an engineer would get wrong by testing naively.

**Model answer:** **Pros** `[T]`: it supports more evaluation types than pairwise (single-output quality, ranking a list); "you don't need to train a specialized model. You could even use an API model"; and it adapts when "task specifications or your preferred outputs change." **The three cons** `[T]`: (1) **Non-determinism** — "they are inconsistent across calls. So most API models are non-deterministic even at temperature zero," with the concrete failure being A-better-than-B on call one and B-better-than-A on call two. (2) **Instability from the environment** — template edits, silent model updates behind an API, and input-order swaps. (3) **Intransitivity** — "rewards are generally not transitive": A beats B, B beats C, but C beats A, because "the model doesn't have a mental model of what came before" and "may be sort of not using the same criteria to grade across calls" `[T]`. **The trap an engineer gets wrong** is which test to run. The lecture is explicit: "you're **not likely to see problems from non-determinism at temperature of zero**. What you're more likely to see is if you **change your instruction template slightly** or the **model updates behind the scenes**… or you **swap the order of the two inputs**" `[T]`. So a temperature-0 determinism A/B proves nothing. The three tests that matter are template perturbation, input-order swap, and a pinned model version. A student's counter-argument — that some rewards are *intrinsically* non-transitive (rock-paper-scissors) — is recorded and partly conceded: "I think that's a solid point… the phenomenon is maybe a little broader than the cases where the reward itself is not transitive, but there's certainly cases where maybe this isn't a property we need to enforce" `[T]`.

**Signal:** Volunteers the naive-test correction unprompted — that a temperature-0 check is the wrong instrument — which is the operational payload of the whole question.
**Follow-ups:**
- *How do you detect non-determinism in production then?* — Order-swap and template-perturbation tests nightly, plus the replay path — Q22.
- *What magnitude of order sensitivity should you expect?* — The transcript gives none `[T]`; measure your own — Q14.
**Red flags:** Says "temperature 0 makes it deterministic"; lists the cons without knowing the test that actually catches them; proposes an ENSEMBLE of judges without addressing that intransitivity makes aggregation ill-defined.

#### T05-Q5 · Verbosity bias and the direction reversal
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** You have internalised length normalisation from search. What is different about reward models, and what is the production consequence?

**Model answer:** The direction reverses. "While pretty much everything we've done so far has had to compensate for **shorter things being lower probability**, **reward models have the opposite problem that longer things tend to be higher reward**" `[T]`. In search, the bias favours short outputs and you normalise *up*; in reward modelling the bias favours long ones and you penalise *down*. The mechanism is in the annotation instructions: "in practice, especially if annotation instructions are not 100% comprehensive, people tend to prefer longer outputs," because "longer outputs annotators tend to describe as more detailed or more comprehensive," and "if you are annotating for **helpfulness**, detailed and comprehensive sound like pretty reasonable things to be" `[T]`. The evidence is a figure cited from an RLHF paper showing "the correlation between the length of the output in tokens and the reward" under "a reasonably strong reward model," with the finding that "even if your outputs are not by any surface attribute better, **even if your outputs are wrong, if they are longer they are like more likely to receive high reward**" `[T]`. **No correlation coefficient is given** `[T]` — measure your own. The correction used in RLHF is "things like applying **length penalties**"; without them "you will see the output space get longer and therefore the reward go up even if the content doesn't necessarily improve" `[T]`. The production consequence: a reranker with no length control will select the wrong answer *because* it is long, which is a worse failure than selecting a boring one. Other demonstrated biases are worth one line: a style/brand prior ("Claude likes headings" `[T]`), annotators voting "really without reading the whole thing" `[T]`, and honest disagreement — a live class vote on two similar outputs landing "maybe 50/50, maybe slightly towards two" `[T]`.

**Signal:** States the reversal as a *sign flip* against the search lesson and can name the annotation-instruction mechanism, not just "models like long answers."
**Follow-ups:**
- *Where do you fix it — training or inference?* — Both have a place; Q17 has the five-option table.
- *How do you know the fix worked?* — Re-measure the score-length correlation; do not assume normalisation removed it — Q23.
**Red flags:** Confuses this with the search-side length bias; claims a penalty "solves" verbosity without a re-measurement; quotes a correlation coefficient.

#### T05-Q6 · Why can a reward model beat the generator at judging?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A reward model is often smaller than the generator it ranks. Why is that not a contradiction?

**Model answer:** Three reasons, and the third is the deep one `[T]`. **Specialisation:** "you can also specifically train a model where the only job is to give you a high score if it's a good output and a low score if it's bad output instead of using the model capacity to both generating and evaluating." A generator spends capacity doing two jobs; a scorer spends all of it doing one. **Whole-output access:** "you can feed in to the model the entire output, or prompt the model beforehand that it should be looking for particular things" — bidirectional access over a complete sequence, which the autoregressive generator never had at the moment it had to commit to each token. **It is the easier problem:** "**It's easier to check at the end than at the beginning**" `[T]`. That third line is the same insight as [T04](./T04-test-time-compute.md)'s observation that things hard for a model to do are hard for it to check, approached from the other side. The concrete failure a checker fixes and a generator cannot: with "a 100-token max" on generation, "two outputs that look like perfectly reasonable for the first 90 some tokens, but one of them finishes within 100 tokens and the other one doesn't" `[T]`. The generator could not know at token 40 which branch would complete; the scorer sees both complete (or truncated) strings and can tell instantly. Hence the combined recommendation: "doing RLHF and then **also doing best-of-N on some small set of outputs** can get you the kind of the **best of both worlds**" `[T]` — training improves the distribution, reranking fixes the specific sample.

**Signal:** Reaches for "it's easier to check at the end than at the beginning" and connects it to the generator's inability to see forward — rather than saying "reward models are just better."
**Follow-ups:**
- *Does that mean a scorer can be arbitrarily small?* — No — it must be better at ranking than the generator is at generating; Q15 puts a floor under it.
- *What is the practical combination?* — RLHF plus best-of-N on a small set — Q19.
**Red flags:** Attributes it to the reward model being "more accurate"; cannot explain why whole-output access is a structural advantage.

#### T05-Q7 · RewardBench v2, ties, and what a leaderboard score does not tell you
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** How do you choose a reward model, and what does the benchmark you would reach for actually guarantee?

**Model answer:** RewardBench v2 is the named reference in this material, and the lecturer highlights two structural things about it `[T]`. It is a **grouping** across evaluation categories rather than a single score, and it is "the first benchmark to include a **notion of ties**" — which is the innovation that matters here, because a benchmark with no tie option forces a strict order on pairs where no meaningful order exists. **What it does not give you:** the lecture describes the benchmark's structure and its tie notion but quotes **no numeric scores** `[T]`. So "pick the top reward model on RewardBench" is not an answer this corpus supports; you would be importing a number from outside it. The reason a benchmark is needed at all is distribution shift: a reward model that looks strong on held-out preference data can collapse on adversarial prompts `[R]` (cheat sheet, from the lecture's framing). Two practical consequences for a selection pipeline. First, the benchmark's tie notion validates the design decision to emit ties rather than manufacture a winner — that decision has to be in the ranker from day one. Second, a public leaderboard measures a *general* preference distribution, and your surface's preference distribution is not that — the lecture's rainbow example ("is green or blue a better answer… by and large these are about the same quality output," expect "somewhere around 0.5" `[T]`) tells you that a third of your comparisons may be near-ties that no leaderboard score predicts. Calibrate on your own annotation set, and treat the benchmark as a screen for candidates rather than a selection.

**Signal:** Knows the two structural facts (grouping, ties) *and* volunteers that no scores are stated — refusing to invent a leaderboard number is the discriminator.
**Follow-ups:**
- *Why does a tie notion in a benchmark change the pipeline?* — It legitimises emitting a tie; a strict-order ranker is manufacturing information — Q15.
- *What is the adversarial-prompt failure?* — The benchmark exists because held-out preference accuracy does not transfer; treat it as the reason to audit on your own adversarial set.
**Red flags:** Quotes a RewardBench v2 score; treats the benchmark as a selection mechanism rather than a screen; does not know a tie category exists.

#### T05-Q8 · Post-hoc content filters: what are they, and what does their placement buy you?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A product manager asks you to "add the safety model to the reward model." Untangle that, and explain why the filter's placement in the pipeline matters.

**Model answer:** They are different components answering different questions, and the lecture draws a hard line. Filters are "**not actually trained into the model**. Generally these are applied **post hoc**. Which is why you can **see sort of outputs that partially cut off**" `[T]`. The definition by contrast is exact: "this is another type of reward model which given an output or a partial output says **not is this a good output or not but is this a safe output**" `[T]`. Placement: "that's **not best of N** — that's sort of like a **post hoc filtering step**." Real examples given `[T]`: a Chinese model describing tourism in Beijing that cut off, characterised as "overdone content filtering"; and a story-generation run where models "write themselves into content moderation holes" — a fancy tattoo on a wrist led through "cut into the wrist" to a self-harm filter. **Why placement is the whole design.** Because the filter is post hoc and independent of the ranking, the pipeline can **fall through to the next-ranked candidate** instead of truncating the chosen one. That is the difference between a customer-visible mid-sentence cut-off and a working system. If you merge the filter into the reward model — one scalar that must encode both safety and quality — you lose the fall-through, because a low score gives you nothing to fall through *to* except the next candidate, which may be identical in the property that failed. So: safety is a separate stage, downstream of ranking, with a fall-through path, and it is never the reward model.

**Signal:** States the fall-through consequence as the reason for the separation — the architectural payload — rather than describing filters as "a safety check we also run."
**Follow-ups:**
- *Are safety filters and guardrails the same thing?* — See [T18](../01-case-studies/T18-guardrails-security.md) for the wider layer; here the point is only placement relative to the ranker.
- *What if every candidate fails the filter?* — Same handling as pool exhaustion — Q24.
**Red flags:** Proposes folding safety into the reward score; does not know filters run post hoc; proposes truncating the output as the fix for a filter rejection.

---

### Mechanism

#### T05-Q9 · The KL bound: state it, orient it correctly, and give me the numbers
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Give me the frequently-cited best-of-N KL bound. Which two distributions does it relate, and what are its values for the `n` you would actually consider?

**Model answer:** The bound is

```
KL(P_bon || P_target) ≤ log n − (n−1)/n
```

and the orientation is the part people get wrong. The lecture says you "draw this sort of **loose upper bound** on the divergence between the **KL divergence between your best-of-N outputs distribution and your target distribution** as **log n minus n minus one over n**" `[T]` — and it clarifies that the target is "the distribution of your target which is sort of the **preference distribution**" `[T]`. So the comparison is between the **best-of-N policy and the target/preference distribution**, *not* between best-of-N and the base policy. Some secondary summaries orient it against `P_base`; that is wrong and it inverts the intuition. Under the correct reading, **the bound and the quantity it bounds both rise with `n`** — more samples buy more aggressive alignment, and the KL is the price you pay for it. The good news is that the price grows only as `log n`. My arithmetic on the bound, `[D]`, with assumptions shown — this is the formula evaluated, not any measurement:

| `n` | `log n` | `(n−1)/n` | bound |
|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.693 | 0.500 | 0.193 |
| 4 | 1.386 | 0.750 | 0.636 |
| 8 | 2.079 | 0.875 | 1.204 |
| 16 | 2.773 | 0.938 | 1.835 |
| 32 | 3.466 | 0.969 | 2.497 |
| 50 | 3.912 | 0.980 | 2.932 |
| 100 | 4.605 | 0.990 | 3.615 |

Two provenance notes. The `n = 1` row giving zero is **my own derivation** `[D]` — the transcript does not state it — and it is consistent with best-of-1 being the generator itself. The transcript also does **not** use the phrase "diminishing returns" `[T]`; the log growth is what that phrase describes, but say it as your own reading.

**Signal:** Orients the bound to `P_target` and can produce the table — and says plainly that it is an upper bound on a price, not a quality dial.
**Follow-ups:**
- *How would you use this in a design review?* — You would not; Q10 is why.
- *What did the lecture's figure actually plot?* — Dotted blue is this bound, red is a much tighter more complex bound, black dots are the empirical KL between the best-of-N policy and the target `[T]` — Q10.
**Red flags:** Orients the bound against the base policy; presents the bound as an exact KL value; cannot produce a single value of the formula.

#### T05-Q10 · The bound is loose, not exact — so what does it guarantee?
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** A staff engineer proposes sizing `n` from the KL bound because "it's a principled upper bound." Take that apart.

**Model answer:** The proposal fails on the lecture's own words. The bound "is often quoted as an **exact value** like this is — this will be the KL divergence, or at least as a **very tight bound**. There's actually some very interesting work in the theory direction which argues that this is **not a tight upper bound for a number of sort of edge cases, some of which do occur in practice**" `[T]`. The paper is attributed as "a really excellent paper by **Barami at all**" — **Beirami et al.; "Barami at all" is an ASR artefact** — and the tighter result is "like **equation 25** in this paper," reached by "a **long derivation**… which I think is quite interesting to read, but we're **not going to go through in depth today**" `[T]`. The figure makes the looseness visible: the dotted blue line is `log n − (n−1)/n`, "this red line here is a much tighter sort of more complex bound," and "this black line here or these black dots here are the empirical KL divergence between a best-of-N policy and the target distribution as `n` grows" `[T]` — and the empirical points sit below the bound. So what the bound guarantees is that **a bound exists and it grows only logarithmically**, which is the useful intuition: best-of-N is a cheap way to move a policy, no gradient step required. It does **not** give you a number you can set `n` to, and it does not tell you where your own quality curve flattens. **The right way to answer a sizing question** is: measure the empirical quality-versus-`n` curve on your own task, then snap `n` to a batch boundary on the flat part of that curve (Q11). The bound is a piece of theory you cite to explain *why* best-of-N works at all; it is not a tuning knob. A candidate who volunteers the looseness before being asked has read the lecture rather than the folklore.

**Signal:** Says "loose, not exact," names Beirami et al. (correcting the ASR), and refuses to size `n` from it — the three-part discriminator.
**Follow-ups:**
- *What does the empirical data in the figure show relative to the bound?* — It sits below; the bound is not tight.
- *What would change your mind about using it?* — Nothing available in this corpus; the lecture gives no usable numeric result from it. The worked usage is illustrative only — Q20.
**Red flags:** Treats `log n − (n−1)/n` as the KL you will actually pay; quotes a specific `n` as "the bound's optimum"; dismisses the caveat as academic.

#### T05-Q11 · Choosing `n`: the hardware argument
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** How do you actually pick `n`? Most people treat it as a quality dial — do better.

**Model answer:** Reach for the hardware argument first, because it is the lecture's own: "you could choose to set the value of `n` to be **something related to your hardware**. Like maybe if you can **generate 32 things in a single batch**, but you would have to use a **second batch or a second pass or a second GPU to generate the 33rd thing**, you could **set `n` to be 32**" `[T]`. That reframes `n` as a **batch-fit parameter**. Going from 32 to 33 does not cost 3% more; it costs an entire additional batch — the cost curve is a **step function**, so every sample below a step boundary is effectively free and the first sample above it is very expensive. The second half of the argument is that `n` is problem-specific: for "what's 2 plus two, you probably **don't need to do like `n` equals 100** and choose from your 100 examples that probably all say something like four" `[T]`. Easy inputs produce near-identical candidates, so the pool's effective size is 1 no matter what `n` says. The process that follows: **measure the empirical quality-versus-`n` curve on your own task**, then set `n` to the largest batch-step boundary sitting on the flat part of that curve. Do not derive `n` from the KL bound (Q10) and do not inherit another team's number — the batch boundary is a property of your deployment shape, and it moves when batch size or GPU count moves. The cheat sheet records the lecture's illustrative typical range as `n = 10` or `100` `[R]`; treat that as a range the lecture mentions, not a recommendation.

**Signal:** Says "step function, not a smooth increment" and connects `n` to the deployment shape — the reframe from quality dial to hardware parameter is the whole answer.
**Follow-ups:**
- *What invalidates your chosen `n`?* — A batch-size or GPU-count change moves the boundary; re-snap rather than re-guess.
- *When would you go below the batch boundary?* — When the measured quality curve has already flattened — drop to the next step down.
**Red flags:** Picks `n = 32` "because the lecture says so" with no batch reasoning; treats `n` as monotone in quality; uses the KL bound to justify a value.

#### T05-Q12 · Generation versus scoring: which side of the pipeline is expensive?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** You have budget for either a bigger generator or a bigger scorer. Which, and why?

**Model answer:** The scorer, and the asymmetry is stated twice in the lecture: "because when you generate you have to **wait for each new token to be generated**, whereas when you're scoring you can **pass everything through as one block**. And also when you generate you're generating **many things**. When you're scoring you're scoring sort of **only a few things**" `[T]`. Unpack it. Generation is `n` independent *autoregressive* decodes, each serial in time and each holding KV for its whole length. Scoring is **one parallel forward pass over all `n` completed sequences** — no decoding loop, no per-token latency, full bidirectional attention over a sequence that already exists. So the marginal cost of a stronger scorer is far below the marginal cost of a stronger generator at the same parameter count. The consequence the lecturer draws is that you can "play around with different variants of using a **smaller model versus larger reward**… and vice versa" `[T]` — and the natural configuration for a reranking pipeline is a **small, fast generator with a disproportionately strong judge**. Note what this does to the design: it means the *quality of the judge*, not the size of `n`, is where the marginal dollar should go, because `n` multiplies a cost you are already paying while judge quality multiplies nothing. **`[D]`** — if you want a rough figure for the case study's arithmetic, scoring at one-eighth the cost of an equivalent generated token gives the `n = 8` Draft Assist configuration a 9x total multiplier rather than 8x; that 1/8 factor is an assumption, not a measured ratio, and the transcript supplies no number at all.

**Signal:** Explains *why* scoring is cheaper (one parallel bidirectional pass versus `n` serial decodes) and draws the design conclusion — invest in the judge, not in `n`.
**Follow-ups:**
- *Then why does anyone increase `n`?* — Because it is the only lever that needs no new model — but it multiplies the expensive term; Q27.
- *Does the asymmetry survive if the judge is an API call?* — No — per-call cost multiplies by `n` with no batching discount, which is why the generative judge is rejected as a production ranker — Q15.
**Red flags:** Assumes the larger model should always generate; cannot say why a parallel pass beats an autoregressive loop; quotes a cost ratio as if measured.

#### T05-Q13 · Diversity: `n` is the sample size, not the candidate set
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Your reranker is configured for `n = 100` and the quality gain has plateaued. Nothing is broken. What is happening, and what is the fix that has a catch?

**Model answer:** You are paying for 100 samples and ranking far fewer distinct candidates. The lecture's illustrative figure: "if you know you're generating a 100 things and you know **temperature sampling with temperature point 2 gives you only maybe 20 unique things on average**, you could try varying temperature" `[T]`. So the candidate set is roughly a fifth of nominal, and a reranker "cannot select what the sampler never produced" — its ceiling is the pool, not the ranker. **`n` is the size of the sample; the effective set is what is unique.** The obvious fix — vary temperature, or sample from several models — has the catch the lecture names: such mixtures "**are sort of not well defined if you are changing what your proposal distribution is** across the process of sampling that set." The example is exact: "**best of 50 but you got 20 of those outputs from Qwen and 10 of those outputs from GPT-5 and 20 of those outputs from Qwen but with like a completely different set of decoding settings**. Now you **no longer have sort of one proposal distribution**" `[T]`. The rejection-sampling framing of Q2 formally breaks. The practical verdict is nevertheless encouraging — "this is **still quite effective in practice**" `[T]` — so the engineering resolution is not to avoid mixtures but to **scope them**: vary temperature only where the *purpose* is diversity, keep the proposal fixed where auditability is required, and record the mixture in the ledger so the assumption break is visible rather than silent. Also measure the number of *unique* candidates per request as a first-class metric; a plateau in quality with a plateau in uniqueness is a sampler problem, and raising `n` will not fix it.

**Signal:** Distinguishes nominal `n` from effective candidate-set size, and volunteers the proposal-distribution break *and* the "still effective in practice" verdict together — the pair is the discriminator.
**Follow-ups:**
- *What do you measure?* — Unique-candidate count per request, and the duplicate rate at the top of the ranking.
- *Does this change your `n`?* — Usually downward; a smaller `n` at better diversity can dominate a larger nominal `n` — Q27.
**Red flags:** Prescribes "raise `n`" or "raise temperature" with no uniqueness measurement; treats multi-temperature sampling as free; does not know the proposal-distribution assumption exists.

#### T05-Q14 · Testing a judge that is a model
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You have to sign off on a generative judge's stability before it goes anywhere near production. Write the test plan, and tell me what the naive version of that plan misses.

**Model answer:** The naive plan is "run it twice at temperature 0 and check the answer matches." The lecture says that plan is close to worthless: "you're **not likely to see problems from non-determinism at temperature of zero**. What you're more likely to see is if you **change your instruction template slightly** or the **model updates behind the scenes** and you don't know this because you're calling an API provider, or you **swap the order of the two inputs**" `[T]`. So **the plan is three perturbation tests, not one determinism test.** (1) **Order swap** — score `(A,B)` and `(B,A)`; the failure signature is A preferred in one and B in the other. This is the one the lecture names twice, and it is not caught by a temperature-0 repeat. (2) **Template perturbation** — whitespace, a reworded instruction sentence, a moved rubric line; the failure signature is a score shift large enough to reorder a close pair. (3) **Version pinning** — record the exact judge version on every call, and re-run a fixed audit set whenever the version changes, because a silent upstream update is invisible in any other instrument. Add a fourth that the lecture implies: **intransitivity**, tested by scoring all three of A, B, C pairwise and checking the tournament is acyclic — "rewards are generally not transitive… if you ask it is A or C better, you're not always going to get the answer that A is better" `[T]`. **What the transcript does not give you is a magnitude** `[T]` — no threshold for "acceptable" order sensitivity or template drift. So you set your own from the audit set's score-gap distribution, and you write it down before you look at the numbers. Any of tests 1–3 failing means the judge is not a production ranker; it is a shadow-mode comparator.

**Signal:** Replaces the naive determinism test with the three named perturbations and adds intransitivity — and explicitly refuses to invent a tolerance the corpus does not supply.
**Follow-ups:**
- *What if only the template test fails?* — Pin the template as a versioned artefact and put its version in the ledger — Q22.
- *Where does the judge go if it fails?* — Shadow mode against the scalar ranker; its agreement rate becomes a monitored metric — Q15.
**Red flags:** "Run it twice and compare" as the whole plan; accepts an order-sensitive judge with a mitigations note; quotes a tolerance figure from nowhere.

---

### Tradeoffs

#### T05-Q15 · Scalar ranker or generative judge: own the decision
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** You need a production ranker for a customer-facing surface with an audit requirement. Choose, and defend the choice against the strongest objection to it.

**Model answer:** Trained scalar reward model, and the audit requirement is what decides it. **The case for the generative judge is real:** no specialised model, can be an API call, adapts when "task specifications or your preferred outputs change" — the lecture's own example is a fact changing (a new president) so the preference ordering should change `[T]`. **Why it still loses.** The requirement is a byte-identical score for a fixed input on replay, and the lecture documents that a generative reward model "is **inconsistent across calls**" and that "**most API models are non-deterministic even at temperature zero**," with the worked failure being A preferred on one call and B on the next `[T]`. On top of that: it breaks under template edits and silent model updates `[T]`, and it is intransitive `[T]`, so a three-way selection has no stable order. A pipeline that ranks 32 drafts with a judge that can flip its verdict has **no audit trail at all** — and an unpinnable judge version voids replay even when the scores happen to agree. **The scalar model's corresponding weakness is also real** and you should state it: it needs preference data and a training loop, it "inherits every bias of its annotation instructions" `[T]`, it is "not well defined over subsequences" `[T]`, and it needs a tie band from day one because some pairs have no meaningful distinction `[T]`. **The strongest objection** is adaptability: when the brand voice shifts, retraining a scalar model is slower than editing a prompt, so the generative judge wins exactly when the spec moves faster than your training loop. **The resolution** is architectural, not a compromise: ship the scalar ranker as the production ranker, run the generative judge in **shadow mode** on a sample, and monitor its agreement with the scalar model. If agreement is high, the generative judge is a cheap drift detector; if it is low, you have found the disagreement before your customers did. Revisit only when the scalar model provably cannot be retrained fast enough — and even then, pin the judge version or it still cannot be the production ranker.

**Signal:** Picks the scalar model *for the audit reason* rather than for accuracy, states the scalar model's own weaknesses without being prompted, and resolves the objection with shadow mode rather than a hedge.
**Follow-ups:**
- *What if the audit requirement were removed?* — The generative judge becomes viable for exploration and low-stakes triage; it is still the weaker ranker and still intransitive.
- *What does the shadow judge actually measure?* — Agreement rate, plus a per-category breakdown that isolates the categories where the two disagree systematically.
**Red flags:** Picks the API judge for convenience; picks the scalar model and claims it has no weaknesses; proposes an ensemble without addressing intransitivity.

#### T05-Q16 · Same-family or cross-family ranker?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Your generator is a Qwen-family model. Does the ranker come from the same family or a different one? There is a benchmark result here — give it, then tell me why it is a trap.

**Model answer:** The benchmark result, from the RewardBench paper, is counter-intuitive and the lecture flags it as such `[T]`: "**even if a model is better on this sort of absolute ranking of preference**, a model that is **from the same model family that you are using to generate the input the outputs is more likely to be a good preference model for that data**… more likely to be a better model to do RLHF with and… for best of N." The intuition offered: "these models have **similar distributions**… your reward model is also likely to place high probability" where the generator does. **Why that is a trap.** A ranker that shares the generator's distribution shares its blind spots. The pushback is recorded in the lecture and the lecturer concedes it substantially: asked whether this is a bias risk from shared blind spots, "I think this is like **certainly possible**… at least partially that like you have a **better warm start**," and "it's definitely possible that like **whatever pathologies you have in your reward model, you're also kind of injecting into the base model**" `[T]`. **No magnitude is given** for the same-family advantage `[T]` — so you cannot price the tradeoff from the corpus. **The resolution is to split the roles rather than pick a side.** Use a same-family ranker, because it genuinely ranks better on this data, and pair it with a **cross-family safety filter and a cross-family audit** — a different family is far less likely to share the generator's characteristic error, so it detects exactly the failure the same-family ranker cannot see. Then audit the ranker's blind spots directly: construct adversarial pairs where the generator is known to be wrong, and check whether the ranker prefers them. If the audit shows the ranker's blind spots overlap the generator's, the fix is to move the ranker cross-family even at a measurable quality cost — a shared pathology is worse than a weaker ranker, because it is invisible from inside the pipeline.

**Signal:** Gives the same-family result *and* the shared-pathology risk, then resolves them by splitting ranker and safety/audit roles rather than picking a winner.
**Follow-ups:**
- *Is the same-family bias ever desirable?* — The lecture's concession is a warm start; it is desirable until the generator has a systematic error, at which point it is the problem — Q16's revisit condition.
- *How do you test for a shared blind spot?* — Adversarial pairs where the generator is known-wrong, scored by the ranker.
**Red flags:** Picks same-family and stops; picks cross-family and sacrifices ranking quality with no cost stated; quotes a same-family advantage magnitude.

#### T05-Q17 · Length control: five options and one default
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** The ranker prefers longer drafts and one of them is now confidently wrong. Give me your options for length control and tell me which one you would ship first.

**Model answer:** Five options, in ascending order of bluntness `[T]`. **Monitor only** — zero cost, does nothing; never alone. **Length-normalised score at ranking time** — divide or adjust the score by token count, adjustable without retraining and directly testable; this is the default at inference. Its weakness is that it assumes a linear relationship, and if the score-length relationship is non-linear you are over- or under-correcting. **Length penalty in training** — the RLHF standard, "things like applying **length penalties**" `[T]`, effective but blunt and requiring a retrain to adjust; too strong and it punishes genuinely detailed answers. **Length band filtering** — reject candidates outside a band *before* ranking; gives a hard guarantee, but throws away good candidates and breaks when the correct answer legitimately needs more tokens. **No length control** — never; the lecture's finding is that "**even if your outputs are wrong, if they are longer they are like more likely to receive high reward**" `[T]`, so an unnormalised ranker will select the wrong answer *for being long*. **What I would ship first:** length-normalised ranking by default, because it needs no retraining, it is testable, and it can be adjusted when the monitor tells you it was wrong. Then the monitor, because the fix is unverified until the correlation is re-measured. Then a band filter for a surface where output length is contractually bounded — long-form help-centre articles rather than chat replies. Then a training-time penalty if the monitor shows the bias survives normalisation, which is the signal that the ranker has *learned* the bias rather than merely inherited it. **The measurement discipline is the answer.** You do not get to say "we normalised, so it is fixed"; you re-measure the score-length correlation on an audit set, and you set a threshold on `|ρ|` before you look. The lecture gives no correlation coefficient and no target `[T]` — the threshold is yours, and stating that it is yours is part of the answer.

**Signal:** Orders the five options by bluntness and picks the inference-time fix first *because it needs no retrain and is reversible*, then insists on re-measurement rather than assuming the fix worked.
**Follow-ups:**
- *What does a surviving correlation mean?* — The model learned the bias; that is a training-time fix — Q23.
- *What if the relationship is non-linear?* — Then linear normalisation is the wrong instrument; fit the relationship, do not assume it.
**Red flags:** Jumps straight to a training-time penalty; claims normalisation fixes it without a re-measurement; quotes a correlation threshold.

#### T05-Q18 · The only quantified reranking result in the corpus
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Give me the one measured production-grade reranking number you know. Then tell me what it does *not* let you conclude.

**Model answer:** The result is from lecture 11, describing a paper the transcript attributes to a collaboration including "**Sweden**" — **an ASR artefact for UC Berkeley; say the correction** `[T]`. The work trained software-engineering agents with RL and "a **critic model to rerank multiple candidate trajectories**" `[T]`. The finding: starting "**from a model that had about 20% accuracy** at the time, **every time you double the number of rollouts you do, you see an approximately constant gain** in the amount that the score increases on SWE-bench. And we were able to get from **around 20 to up to 32** here" `[T]`. Cost: "run inference like **16 times**" `[T]` — 16x for a 1.6x relative improvement in accuracy, on a task where "running agents is expensive already." The **shape** of the claim matters more than the number: **log-linear gains per doubling over the tested range, not flattening** `[T]`. The lecture does not report where it flattens. **What it does not let you conclude.** First, it is a different task — SWE-bench agent trajectories with a trained critic — so it is not a prediction for support drafts or help-centre articles. Second, the lecture's own viability framing is a per-task budget: "let's say you were working on an agent to run like a machine learning experiment… and you could spend **$10,000** on making sure that it succeeded, then this could be a viable option" `[T]` — 16x inference is viable for a high-value task, not for a 1.2M-drafts-a-month surface. Third, there is a **vendor claim** adjacent to it that must be flagged as such: the **Claude Sonnet 4.5** launch chart where "the **dark bar is single instance inference** and the **light bar is where they did lots of rollouts and then they reranked them**" is the lecturer's characterisation of "what everybody does" to win a leaderboard `[T]` — that is a **vendor's published evaluation, not an independent measurement**. If you quote a reranking gain, quote this one with its cost, its task, and its provenance attached.

**Signal:** Quotes 20% → 32% *with* the 16x cost and the "approximately constant per doubling" shape, and refuses to generalise it across tasks — plus flags the adjacent vendor number.
**Follow-ups:**
- *How do you use this to justify your own `n`?* — You do not; you measure your own curve — Q11. Any interpolation to your surface is your arithmetic, labelled `[D]`.
- *Outcome or process reward model here?* — Outcome — "you evaluate the roll-out using unit tests… you train a model to predict whether the output was correct or not. So this is the simplest way… this is an **outcome reward model**" `[T]`; a process model needs supervision that is "much more complex" `[T]`.
**Red flags:** Quotes 20→32 without the 16x cost; presents it as a general best-of-N law; cites the Sonnet 4.5 chart as evidence rather than a vendor claim.

#### T05-Q19 · RLHF plus best-of-N
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You already train the generator with RLHF. Why would you also pay for best-of-N, and what does each one fix that the other cannot?

**Model answer:** Because they operate on different objects, and the lecture's own recommendation is the combination: "doing RLHF and then **also doing best-of-N on some small set of outputs** can get you the kind of the **best of both worlds**" `[T]`. **RLHF changes the distribution; best-of-N fixes the sample.** Training moves the generator so that the *average* sample is better — it improves everything the model produces, permanently, at zero inference cost. Reranking cannot do that: it can only choose among what the sampler produced, so its ceiling is the pool. **Best-of-N fixes what training cannot reach.** The lecture's concrete failure is a length-capped generation: with "a 100-token max," two outputs "look perfectly reasonable for the first **90 some tokens**, but one of them finishes within 100 tokens and the other one doesn't" `[T]`. The generator could not know at token 40 which branch would complete — a distribution-level improvement does not fix a per-sample truncation accident. A scorer looking at the completed candidate can. **The general principle** is the same as Q6's third reason: "it's easier to check at the end than at the beginning" `[T]`, so a check-and-select pass catches a class of error that a training pass structurally cannot. **What this means for the design:** the two are complements in the cost model, not substitutes. RLHF is a fixed training cost amortised over all requests; best-of-N is a per-request multiplier on generation. So best-of-N earns its place on the *low-volume, high-stakes* surface, where a 16x multiplier is affordable and the marginal correctness matters — not on the high-volume surface, where training is the only lever whose cost per request is zero. **The case study's scope caveat carries here:** the corpus states that "RLHF is not covered much in this class" `[T]`, so this answer is about the architecture of the combination, not about how to run the training.

**Signal:** Says "training changes the distribution, reranking fixes the sample" and places each on the right surface by cost structure — rather than calling them two ways to do the same thing.
**Follow-ups:**
- *Which surface gets which?* — Best-of-N on low-volume high-stakes; training on high-volume — Q27.
- *Does reranking substitute for RL?* — It is a bounded policy improvement, capped by `log n` and by the pool; it is not a training replacement — Q9.
**Red flags:** Treats them as alternatives; claims best-of-N makes RLHF unnecessary; cannot name a failure training cannot fix.

#### T05-Q20 · Two different `n`s: 1,000 inputs at `n = 50` versus `n = 32`
**Difficulty:** L5 · **Depth expected:** 5 min
**Question:** I am going to say two numbers from this material: one thousand, and fifty, and thirty-two. Tell me what each is doing, and what a candidate who conflates them would get wrong.

**Model answer:** They are three different roles and conflating them is a real error. **The `n = 32` is a hardware batch-fit parameter.** It is the lecture's own rationale: "maybe if you can **generate 32 things in a single batch**, but you would have to use a **second batch or a second pass or a second GPU** to generate the 33rd thing, you could **set `n` to be 32**" `[T]`. Its value comes from the deployment shape — batch size and GPU count — and it changes when those change. **The 1,000 and the 50 belong together, and they are the lecture's illustrative use of the KL bound, not a recommendation.** The statement is: "if you had say a **thousand inputs** and for each of them you did best of `n` with **`n` equals 50**, you could bound the KL divergence between the distribution of your best output from each of those and the distribution of your target, which is sort of the **preference distribution**" `[T]`. Here **1,000 is the number of *inputs* over which the divergence is aggregated** — it is the sample over prompts — and **50 is the per-input `n`**, chosen illustratively to demonstrate the bound. **What the conflation gets wrong:** treating 32 and 50 as competing recommendations for the same quantity, or treating 1,000 as a sample size you should generate per prompt. They are not on the same axis. 32 answers "how many candidates fit in one batch"; 50 and 1,000 answer "what does the bound look like when you aggregate over a thousand prompts at fifty samples each." A candidate who says "the lecture recommends `n` between 32 and 50" has merged a hardware argument with an illustrative bound evaluation and produced a recommendation the corpus does not contain. **Also worth stating:** the lecture "**did not compute a numeric result**" from the 1,000 × 50 illustration `[T]` — it is a bound demonstration, and the case study records it as illustrative with no numeric result. If you want a number from the bound, you evaluate the formula yourself and label it `[D]` — that is Q9's table.

**Signal:** Separates the batch-fit parameter from the bound's illustration on the correct axes (per-input `n` versus number of inputs), and refuses to manufacture a recommended `n` from the pair.
**Follow-ups:**
- *So what `n` would you ship?* — The batch-fit answer plus your own measured curve — Q11.
- *What is the 1,000 actually for in the illustration?* — Aggregating the divergence over many inputs, so the bound is a statement about a distribution rather than one prompt.
**Red flags:** Quotes "`n` between 32 and 50"; treats 1,000 as a per-prompt sample size; claims the lecture computed a numeric KL from the illustration.

#### T05-Q21 · Intransitivity: how do you rank when the comparator has no total order?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Your judge says A beats B and B beats C, then says C beats A. You cannot replace it this quarter. What do you do?

**Model answer:** First, correctly classify it. This is **expected behaviour for a generative judge, not an anomaly** — "rewards are generally not transitive… if you ask it is A or C better, you're not always going to get the answer that A is better" `[T]` — and the mechanism is that "the model doesn't have a mental model of what came before" and "may be sort of not using the same criteria to grade across calls" `[T]`. A student's counter-argument is also recorded and partly accepted: some rewards are *intrinsically* non-transitive — "rock paper scissors is not trans[itive]… and if you encode that into a reward model it might not necessarily catch it" — and the lecturer concedes "I think that's a solid point… there's certainly cases where maybe this isn't a property we need to enforce" `[T]`. So do not assume all intransitivity is a defect; ask whether your task's preference is genuinely cyclic. **The engineering fix is to stop doing pairwise comparisons.** Score every candidate against a **fixed reference** rather than against each other, so each score is anchored to an absolute point and the ranking is the sort of a vector rather than the closure of a tournament. This is the same move as length normalisation — you remove a degree of freedom (the comparator's context) that the model is not using consistently. If a fixed reference is not available, score against a small fixed set of references and average, which makes each candidate's context identical across calls. **What you accept by doing this:** you lose information — pairwise comparisons are cheaper and can be more sensitive than absolute scoring. **What you refuse to do:** run a sorting algorithm over a cyclic comparator and ship whatever order it produced. A ranker that fabricates a total order from a cyclic judge is manufacturing information, and the symptom is that the winner changes with comparison order, which is exactly the failure the test plan has to catch.

**Signal:** Classifies it as expected, then fixes it by changing the comparison *structure* (fixed reference) rather than by retrying or averaging — and acknowledges the lecturer's concession that some cyclicity is intrinsic.
**Follow-ups:**
- *How do you test for it?* — Score a triple pairwise and check for cycles; run it nightly — Q14.
- *Why does a fixed reference help?* — It makes the model's context identical across calls, which is the variable the judge is not holding constant.
**Red flags:** Calls it a bug and retries until it agrees; sorts with a pairwise comparator and ships the result; does not know a fixed-reference alternative exists.

---

### Debugging

#### T05-Q22 · The ranking is not reproducible on replay
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Legal pulls a selection from three weeks ago and asks why draft 4 won. You rerun it and get draft 6. Nothing was deployed. Walk me through the diagnosis.

**Model answer:** Work the decision tree in the order that eliminates whole branches. **First question: is the ranker generative or scalar?** If generative, "same input, different verdict twice" is documented behaviour — the lecture states such models "are **inconsistent across calls**" and that "**most API models are non-deterministic even at temperature zero**" `[T]`. If that is the answer, it is an **architectural defect, not an incident**: the fix is to restore the scalar ranker, and the incident write-up says the pipeline shipped a ranker that could not be replayed. **If the ranker is scalar, the nondeterminism is yours and it is investigable.** Three suspects. (i) **Batching or precision differences** — the scalar score is a forward pass, so a different batch composition or a different kernel can move the score in the last bits, and a tie-band boundary can then flip the order. Check whether the two drafts were within the tie band; if they were, the "disagreement" is a tie and the ranker should have emitted a tie. (ii) **The judge version moved** — a model update behind the scenes, with the ranking shifting and no deploy recorded. This is the second named source of instability `[T]` and the reason the ledger records a **pinned judge version**; if the version field is empty or unpinned, that is the finding. (iii) **The candidate pool differs** — the generator itself is non-deterministic at temperature 0 — GPU reduction order, multi-GPU aggregation, MoE routing and quantization all perturb the argmax `[T]` (CMU lecture 2) — so the rerun produced a *different pool*, and the ranker faithfully selected the best of a different set. **This third one is the case teams miss**, and it is why the ledger must record the candidate identities, not just the scores. **What the ledger needs to have contained** all along: `n`, every score, the judge version, the seed or sampling parameters, the candidate contents, and the runner-up. Without the candidate pool recorded, you cannot distinguish (iii) from (i), and the audit answer degrades to "we believe it was a tie."

**Signal:** Splits on the ranker type first, then separates *ranker* nondeterminism from *generator* nondeterminism — the pool-differs branch is the one that separates strong from adequate.
**Follow-ups:**
- *What if the judge was an API model with no pinnable version?* — It may not be the production ranker at all; that is a §5.1 decision — Q26.
- *How does the tie band change the answer?* — A gap inside the band is a tie, not a disagreement; shipping a tie is the correct replay output.
**Red flags:** Reruns until it matches; blames the model update without checking the pool; does not know that a scalar ranker can be non-reproducible for batching reasons.

#### T05-Q23 · Outputs are getting longer and nobody changed a prompt
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Acceptance rate is flat, but average draft length is up 22% over two months and the agents are complaining. Where do you look?

**Model answer:** This is the named bias and the direction matters: reward models "have the opposite problem that **longer things tend to be higher reward**," and "**even if your outputs are wrong, if they are longer they are like more likely to receive high reward**" `[T]`. Flat acceptance with rising length is the *signature* — the ranker is not making drafts better, it is making them longer, which is exactly what the lecture says happens when the length term is unmanaged: "you will see the **output space get longer and therefore the reward go up even if the content doesn't necessarily improve**" `[T]`. **Diagnosis, in order.** (1) **Measure the score-length correlation** on a recent audit set. This is the decisive instrument; if `|ρ|` is above your threshold the cause is identified, and if it is not, the cause is elsewhere and you have saved yourself a retrain. (2) **Check whether the bias is inherited or learned.** Inherited means the model was trained on preference data where annotators preferred longer outputs — "if you are annotating for **helpfulness**, detailed and comprehensive sound like pretty reasonable things to be" `[T]` — and inference-time normalisation should cancel it. Learned means the ranker has drifted toward length through retraining, and inference-time normalisation will not fully fix it. The test is whether normalising the score at ranking time restores the old length distribution; if it does not, you are in the second case. (3) **Check the training set's own length distribution** if a retrain happened in the window. **Fix, escalating:** renormalise the ranking score first, because it needs no retrain and is reversible. If the correlation survives normalisation, add a **length penalty in training** — the RLHF standard `[T]` — and re-measure. Only then consider a length band filter, which gives a hard guarantee but throws away good candidates. **The trap to name out loud:** do not raise the acceptance-rate target as the fix; acceptance being flat is the evidence that length, not quality, moved.

**Signal:** Reads "flat acceptance + rising length" as the verbosity signature, reaches for the correlation as the decisive instrument, and distinguishes an inherited bias from a learned one before choosing the fix.
**Follow-ups:**
- *Why is flat acceptance diagnostic?* — Because a real quality gain would move acceptance; length without acceptance is the bias — Q5.
- *What if the correlation is low?* — Then the cause is not the reward model; check the generator's prompt and the tokenizer, and re-check the pool diversity.
**Red flags:** "Add a length penalty" as step one; treats longer output as better quality; does not measure the correlation before changing anything.

#### T05-Q24 · All `n` candidates are bad
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Your ranker's top score on a ticket is far below the accept threshold — and so is every other candidate's. What is the pipeline's contract, and what do you change?

**Model answer:** The contract is explicit and it is not "return nothing": with a bounded `n` we return "the sort of **least bad of our samples**" `[T]`. That is the accepted trade for not sampling forever — the rejection-sampling framing of Q2 is only viable because we want relative behaviour and `n` is fixed, so exhaustion is a designed outcome rather than a bug. **But do not silently ship it.** The important move is to classify the event correctly: **pool exhaustion is a generator signal, not a ranker signal.** The ranker did its job; it ranked a bad pool and told you so. Reranking "cannot create information that is not in the candidate pool" — this is the single tradeoff to volunteer before being asked. So the response is: route to a human, record that the pool was exhausted, and treat **exhaustion rate as a first-class metric** rather than a log line. Then diagnose the generator, in this order: (1) **diversity** — `n = 100` at temperature 0.2 gives "only maybe 20 unique things on average" `[T]`, so if the unique count is low the pool was never really `n` candidates wide (Q13); (2) **task fit** — a request outside the generator's competence will produce uniformly bad candidates no matter how many you draw; (3) **the input itself** — a malformed ticket, a missing attachment, an ambiguous ask; a human routing rule belongs here rather than a model fix. **What you must not do:** raise `n`. More samples from a generator that cannot produce a good answer for this input produces more bad candidates at higher cost. And do not lower the accept threshold — that converts a visible failure into an invisible one, which is the actual dangerous outcome. **The one exception worth naming:** if exhaustion is concentrated on a *narrow* input class, that is a routing problem, not a generation problem, and the fix is a difficulty router rather than a bigger `n` — see [T14](../01-case-studies/T14-routing-gateways.md).

**Signal:** Returns the least-bad candidate *and* refuses to hide it, classifies exhaustion as a generator signal, and explicitly rejects raising `n` or lowering the threshold as fixes.
**Follow-ups:**
- *Why not raise `n`?* — It multiplies the expensive term and adds samples from the same inadequate distribution — Q12.
- *What if exhaustion is uniform across all inputs?* — Then the generator, not the input class, is the bottleneck; consider a model change rather than more samples.
**Red flags:** Lowers the accept threshold to make the alert go away; raises `n` as the reflex fix; does not record the exhaustion at all.

#### T05-Q25 · A vendor reports a big best-of-N gain
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** A model vendor's launch post shows a large accuracy jump from "reranking multiple samples." Your CTO wants to adopt it. What do you ask for, and what would you refuse to conclude?

**Model answer:** First, name what the number is. The corpus contains exactly this pattern: the **Claude Sonnet 4.5** launch, where "the **dark bar is single instance inference** and the **light bar is where they did lots of rollouts and then they reranked them**," described by the lecturer as "what everybody does when they want to beat" a competitor `[T]`. That is **the lecturer's characterisation of a vendor's published evaluation, not an independent measurement** `[T]` — mark it as a vendor claim every time it comes up, and say so out loud in the meeting. **Four questions to ask, all of which the post usually leaves unanswered.** (1) **What is `n`?** A reranked bar with no `n` is uninterpretable — the cost is linear in rollouts and the gain is log-linear per doubling, so `n` is half the claim (Q18). (2) **What is the verifier, and is it the same model that generated the candidates?** A verifier that is the generator, or a same-family model, shares its blind spots (Q16) and the gain will not transfer to your generator. (3) **Is the verifier a rule-based checker or a learned model?** For code, unit tests are the strongest verifier that exists `[R]` (cheat sheet) and the lecture's own example uses unit tests as the supervision signal for an outcome reward model `[T]` — a benchmark-scored gain from an exact checker is not a gain you can expect from a learned judge on an open-ended surface. (4) **What is the cost multiplier?** "At the cost of having to run inference like **16 times**" `[T]` is the corpus's own example, and the lecturer's viability framing is a ~$10,000 per-task budget `[T]` — a high-value agent task, not a 1.2M-requests-a-month surface. **What I would refuse to conclude:** that the gain is available to me at my volume, on my task, with my verifier. The transferable claim is that **reranking with a trained critic can produce approximately constant gains per doubling of rollouts over the tested range** `[T]`; whether that holds for your surface is a measurement you run yourself (Q11). The honest internal answer is: it is worth a two-week shadow evaluation on one surface, with the vendor's `n` and verifier as the starting hypothesis — not a roadmap commitment.

**Signal:** Labels it a vendor claim immediately, asks for `n`, the verifier's identity and family, and the cost multiplier, and refuses to transfer the result to a different task and volume.
**Follow-ups:**
- *What if they will not disclose the verifier?* — Then the claim cannot be replicated and it is marketing, not evidence; run your own shadow test.
- *What would a credible vendor number look like?* — `n`, the verifier's identity, the cost multiplier, and a single-instance baseline measured on the same harness.
**Red flags:** Accepts the gain as a planning input; asks only about cost; does not ask what the verifier is.

---

### Scale and design

#### T05-Q26 · The audit posture for a selection decision
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** A regulator asks you to explain why a particular draft was chosen over the alternatives. What did you have to have recorded, and what breaks the record?

**Model answer:** Start from what "defensible" can honestly mean here. Preference data encodes "whoever annotated the data or more upstream **whoever wrote the annotation instructions**" `[T]`, and a live vote in the lecture on two similar outputs landed "**maybe 50/50, maybe slightly towards two**" `[T]`. So "defensible" cannot mean "objectively correct" — it means **"we can say who decided, and on what basis."** That reframes the requirement: you are not proving the choice was right, you are proving it was made by a recorded process. **The record is a selection ledger**, and the case study's four rungs are the ladder `[R]`: no record (undefensible); the chosen output only (cannot reconstruct the choice); full record (`n`, all scores, judge version, seed, and **the runner-up**) which supports replay, calibration, and drift detection; and the full record plus periodic human re-scoring of the archive, which is what detects slow drift. The fourth rung is the one that turns the ledger from an audit artefact into a monitoring instrument, and it is cheap at a sampled rate. **What breaks the record — and this is the answer's substance.** (i) **An unpinnable judge version.** If the ranker is an API model whose version you do not control, a silent upstream update **voids replay entirely**, and no amount of logging compensates. The case study is explicit that this is reason enough for it not to be the production ranker. (ii) **A missing candidate pool.** Recording scores without the candidates means you cannot distinguish "the ranker changed its mind" from "the generator produced a different pool" — Q22's third branch. (iii) **An unrecorded template version.** Template edits are a named source of score instability `[T]`; the template is a versioned artefact and its version belongs in the ledger next to the judge version. (iv) **No redaction policy for rejected candidates.** Rejected drafts may contain customer PII; the ledger must define what is stored, for how long, and in what form, or the audit trail becomes a liability. (v) **Recording only the winner** — the runner-up is what makes the decision reconstructable, because it is the evidence that a comparison happened at all. **The organisational consequence:** the ledger must ship in the same release as the ranker. A reranker without a replay path is unauditable, and retrofitting the record after the first regulatory request is not possible — the data is gone.

**Signal:** Redefines "defensible" as "we can say who decided," then names the record-breaking conditions — especially the unpinnable judge version and the missing candidate pool — rather than listing log fields.
**Follow-ups:**
- *What does the monthly re-scoring actually detect?* — Slow drift in the ranker or the preference distribution, which no per-request check can see.
- *What is the cost of the full ledger?* — Storage plus a redaction policy; the case study treats it as the default rather than an option — §5.5.
**Red flags:** Lists log fields with no failure conditions; assumes the audit proves correctness; omits the judge version pin as a precondition.

#### T05-Q27 · At 10x, where does the marginal dollar go?
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Traffic grows 10x. Your reranking pipeline has a fixed budget per request. What changes about where you spend?

**Model answer:** **The bottleneck moves from scoring to generation, and the response is a better judge rather than a larger `n`.** The lecture's asymmetry is the whole argument: "when you generate you have to **wait for each new token to be generated**, whereas when you're scoring you can **pass everything through as one block**" `[T]`. At 10x, `n` stops being free — the batch step that made `n = 32` a rounding error at 9k articles a month is a real line item at 12M tickets a month. So the levers reorder. **(1) The judge becomes a service with its own SLO** — pinned versions, calibration windows, canary deploys, rollback — the same discipline as any other model in the stack. This is the case study's own 10x list `[R]` and it is not optional, because the judge is now the component whose quality bounds the system. **(2) `n` becomes adaptive rather than fixed.** The lecture's observation that "what's 2 plus two" does not need `n = 100` `[T]` is a cost argument at this volume: easy inputs produce near-identical candidates, so the effective pool is 1 and the other 31 samples are pure waste. Difficulty-based adaptive `n` moves from nice-to-have to mandatory — see [T14](../01-case-studies/T14-routing-gateways.md). **(3) The generation/scoring configuration inverts.** "The biggest model that fits" stops being the right generator: with a strong judge, "a smaller generator with a larger candidate pool wins" `[T]`, which is the same small-model-plus-more-inference economics as [T04](./T04-test-time-compute.md) applied to selection rather than reasoning. **(4) The annotation pipeline becomes the constraint** — preference data's provenance has shifted from crowdsourcing to expert annotators over time `[T]`, and at 10x the volume of preference data needed to keep the ranker current is a supply problem, not an ML problem. **(5) What survives unchanged:** sample-then-rank with a non-likelihood scorer; the tie band; length normalisation; and the ledger. Those are structural and do not move with scale. **What I would refuse to do at 10x:** raise `n` as the response to a quality complaint. It multiplies the expensive term to fix a problem that is usually the judge's (Q22, Q23).

**Signal:** Says "spend on the judge, not on `n`" and gives the generation/scoring asymmetry as the reason — then names the annotation supply chain as the real scaling constraint, which is the non-obvious answer.
**Follow-ups:**
- *Why does the judge need an SLO at 10x but not at 1x?* — Because it is now the bounding component; a judge regression is a whole-surface regression — Q26's drift detection.
- *What inverts about the generator choice?* — Smaller generator plus more candidates beats the biggest generator once the judge is strong.
**Red flags:** "Scale the GPUs" or "raise `n`"; treats the judge as a static dependency; no view on the preference-data supply.

#### T05-Q28 · Design the reranking pipeline for a new surface
**Difficulty:** L5 · **Depth expected:** whiteboard
**Question:** A new surface drafts internal policy documents — 40k documents a month, read by employees, sometimes quoted in legal proceedings. Higher stakes than chat, lower volume than support, and the drafts must be defensible. Design the reranking pipeline and tell me what you would not do.

**Model answer:** Start from the surface's real requirement, which is not "good text" but **reconstructability**: a quoted document must be traceable to a decision. That single requirement sets most of the design. **Ranker:** a trained scalar reward model — deterministic and replayable, one forward pass over all `n` candidates in parallel `[T]`, no API dependency, and the log-space identity makes it cheap to train from a classifier head `[T]`. The generative judge is rejected as the production ranker for exactly the reason the surface names: a judge that "is inconsistent across calls," and that "most API models are non-deterministic even at temperature zero," cannot support a document quoted in a legal proceeding `[T]`. It runs in shadow mode and its agreement rate is a monitored metric. **`n`:** hardware-fit, snapped to the largest batch boundary on the flat part of the measured quality curve (Q11) — not derived from the KL bound (Q10), and not copied from the support pipeline's value. Low volume means the multiplier is affordable, so the bias is toward the larger boundary. **Length:** length-normalised ranking plus a monitored score-length correlation, with a band filter because internal policy documents have a natural length range and the bias direction here is "longer = more reward" `[T]`. **Safety:** a post-hoc filter as a **separate stage with fall-through to the next-ranked candidate, never truncation** — a truncated policy document is worse than a rejected draft, and the fall-through is what the post-hoc placement buys (Q8). **Judge provenance:** same-family ranker for ranking quality, cross-family safety filter and cross-family audit for blind-spot detection, plus a human gold set at a fixed sample rate for calibration — and near-tie disagreement between the ranker and the annotators is recorded as **data about the task**, since the lecture's own vote landed at "maybe 50/50" `[T]`. **Ledger:** full ledger from day one — `n`, all scores, pinned judge version, template version, seed, the candidate contents, and the runner-up — plus monthly human re-scoring of a sample. A reranker without replay is unauditable, and this surface is the one where that matters most. **What I would not do:** I would not use an API judge as the production ranker; I would not omit the tie band (some policy pairs have no meaningful distinction, and the rainbow example's expected ~0.5 `[T]` says so); I would not score partial drafts, because reward models are "not well defined over subsequences" `[T]` and the reward curve over prefixes "goes negative at several points" `[T]`; I would not raise `n` to respond to a quality complaint; and I would not ship without the ledger in the same release.

**Signal:** Derives the whole design from "reconstructability," commits to a scalar ranker for the legal reason, and prices `n`, length, safety and provenance as separate decisions with named rejections.
**Follow-ups:**
- *What changes if the surface grows to 400k documents a month?* — The multiplier stops being affordable; adaptive `n` and a smaller generator with a stronger judge become mandatory — Q27.
- *What is the first thing you would measure?* — The empirical quality-versus-`n` curve, and the score-length correlation — before setting any threshold.
**Red flags:** Uses an API judge because the volume is low; omits the ledger until asked; scores partial drafts; derives `n` from the KL bound.

---

## Whiteboard exercises

### Exercise 1 — Choose `n` for a new surface, with the KL bound on the table
**Prompt.** `Contract Review` is a new surface: 3,000 requests a month, each generating a clause-by-clause analysis that a lawyer reads. Your serving configuration fits 16 sequences in one batch on one GPU; the 17th requires a second pass. A staff engineer has put the KL bound `log n − (n−1)/n` on the whiteboard and argues for the largest `n` the bound allows. You have 25 minutes. Produce the decision and the reasoning.

**What to produce.** The bound's values at the candidate `n`s, the direction of the bound relative to the quantity it bounds, the reason it cannot size `n`, the sizing rule you actually use, and the measurement that would confirm your choice.

**Expected whiteboard.**

```
The bound, evaluated [D]:
  n=8   log 8   - 7/8  = 2.079 - 0.875 = 1.204
  n=16  log 16  - 15/16 = 2.773 - 0.938 = 1.835
  n=32  log 32  - 31/32 = 3.466 - 0.969 = 2.497

Orientation:  KL(P_bon || P_target) <= log n - (n-1)/n
  P_target is the PREFERENCE distribution, not the base policy.
  Bound and bounded quantity BOTH rise with n -> the KL is the PRICE
  of more aggressive alignment, and it grows only as log n.

Why it cannot size n [T]:
  - "often quoted as an exact value ... or at least as a very tight bound"
  - "not a tight upper bound for a number of edge cases some of which
     do occur in practice" (Beirami et al.; tighter bound = eq. 25)
  - in the figure, empirical KL sits BELOW the dotted bound line

Sizing rule:
  1. measure the empirical quality-vs-n curve on THIS task
  2. batch step here is 16 (the 17th costs a second pass)
  3. n = 16 (largest step boundary on the flat part of the curve)

Measurement to confirm: quality at n = 8 vs 16 vs 32 on a held-out set,
plus unique-candidate count per request (guards against a pool that is
nominally 16 and effectively 3)
```

**Grading rubric.**
- Writes the bound as `KL(P_bon || P_target)` — oriented against the **target/preference** distribution — and does not orient it against the base policy.
- Says the bound is **loose** and quotes the lecture's warning that it is commonly misquoted as exact, naming Beirami et al. (correcting the "Barami at all" ASR rendering) as the source of the tighter equation-25 result.
- Refuses to size `n` from the bound and instead snaps to the **batch-step boundary** (16 here) on the measured quality curve.
- Volunteers that nominal `n` may exceed the effective candidate set, and proposes measuring unique candidates per request.

### Exercise 2 — The ranker is not reproducible, and the surface is audited
**Prompt.** `Draft Assist` selects one of 8 drafts using a generative judge behind an API. Legal pulls a selection from last month, asks why draft 3 won, and your rerun selects draft 6. The API provider confirms no outage. You have 25 minutes to produce the root-cause analysis, the immediate mitigation, and the structural fix.

**What to produce.** A ranked hypothesis list with the measurement that confirms or kills each, the ledger fields that would have made the answer definitive, the immediate mitigation, and the structural change.

**Expected whiteboard.**

```
Hypotheses (ranked by likelihood x cheapness to test)
  1. Generative judge non-determinism .......... test: score the same pair twice (A,B) and (B,A)
        -> documented: "inconsistent across calls"; non-deterministic even at T=0 [T]
        -> NOT caught by a temperature-0 repeat test [T]
  2. Silent judge version update ............... test: judge version field in the ledger
        -> unpinned API version voids replay [T]
  3. Template edit ............................. test: template version field
  4. Candidate pool differed ................... test: were candidate CONTENTS recorded?
        -> generator is itself non-deterministic at T=0 [T] (CMU lecture 2):
           GPU reduction order, multi-GPU aggregation, MoE routing, quantization
  5. Scalar-vs-batch precision ................. N/A here (judge is generative)

Ledger fields that would settle it:
  n | all candidate CONTENTS | all scores | judge version | template version
  | seed / sampling params | runner-up

Immediate mitigation:  restore the deterministic scalar ranker for this surface
Structural fix:        generative judge -> shadow mode only; agreement rate
                       becomes a monitored metric; pin the version or drop it
```

**Grading rubric.**
- Names the three documented instability sources (non-determinism across calls, template edits / silent model updates, input-order swaps) and states that a temperature-0 repeat test does **not** catch them.
- Includes the **candidate-pool-differs** hypothesis — the generator's own nondeterminism means the rerun ranked a different pool — which is the branch most candidates miss.
- Names the ledger fields that would have made the diagnosis definitive, including candidate contents, judge version and template version.
- Gives a structural fix that is architectural (scalar ranker in production, generative judge in shadow mode) rather than "add a retry."

### Exercise 3 — Choose a ranker and an `n` for two surfaces with opposite economics
**Prompt.** Two surfaces. (a) `KB Writer`: 9k articles a month, published once and read for years, each article ~900 tokens. (b) `Draft Assist`: 1.2M drafts a month, ~180 tokens, agents edit most of them. Both currently return the first sample from a temperature-0.7 decode. Pick a ranker for each, pick `n` for each, and say what you would measure to know you were wrong.

**What to produce.** A decision table with the chosen ranker, the rejected alternative, the chosen `n` and its justification, and the wrong-if signal per surface — plus the generation-versus-scoring cost reasoning that makes the two surfaces different.

**Expected whiteboard.**

| | (a) KB Writer — 9k/mo, high stakes | (b) Draft Assist — 1.2M/mo, high volume |
|---|---|---|
| Ranker | trained scalar RM (audit + determinism) | trained scalar RM (same reason) |
| Reject | generative judge — "inconsistent across calls… non-deterministic even at temperature zero" `[T]` | API judge — per-call cost × `n`, no batching discount, no determinism |
| `n` | **32** — the batch-fit boundary; low volume makes the multiplier cheap `[T]` | **8** — larger `n` pushes the cost multiplier past budget |
| Length | normalise + **band filter** (contractual length range) | normalise + monitor the correlation |
| Safety | post-hoc filter, **fall through** to next-ranked, never truncate `[T]` | same |

```
Why the two n's differ [D]:
  generation is the expensive term: n serial autoregressive decodes,
  each holding KV for its whole length [T]
  scoring is ONE parallel forward pass over n completed sequences [T]

  (a) 9k x 32 x 900 tokens   ~ 259M tokens/month   -- negligible
  (b) 1.2M x 8 x 180 tokens  ~ 1728M tokens/month  -- the cost driver
  -> the surface with the highest cost per output has the LOWEST volume,
     so it carries the largest n. This is the asymmetry to state.

Wrong-if signals:
  (a) length correlation |rho| over threshold after normalisation      -> retrain with penalty
  (b) cost multiplier above budget, or pool exhaustion rate rising     -> re-snap n / fix generator
  both: clear-case disagreement with human audit rising                -> ranker defect, not near-tie
```

**Grading rubric.**
- Chooses the scalar ranker for both surfaces *for the audit/determinism reason*, not for accuracy, and rejects the generative judge with the transcript's own non-determinism language.
- Justifies the two different `n` values from **batch fit plus volume**, and states the asymmetry explicitly: the highest cost-per-output surface has the lowest volume, so it carries the largest `n`.
- Gives the generation-versus-scoring cost asymmetry as the reason the multiplier is affordable at all, rather than treating `n` as a quality dial.
- Names a measurable wrong-if signal per surface — at minimum a length-correlation threshold and a cost or pool-exhaustion trigger — and separates clear-case audit disagreement from near-tie disagreement.

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt` — the "rank by something other than log probability" requirement and the mode-seeking prohibition; rejection sampling with the proposal/target/acceptance formalism and the contrast with constrained decoding; the `log n − (n−1)/n` KL bound oriented against the **target/preference** distribution, the "often quoted as an exact value" caveat, and the Beirami et al. attribution rendered as "Barami at all" with the tighter bound as equation 25; the 1,000-inputs-at-`n = 50` illustrative usage; the hardware rationale for `n = 32` and the "2 plus two" counter-example; the generation-versus-scoring cost asymmetry; the temperature-0.2 → ~20-unique-of-100 diversity figure and the broken single-proposal assumption; Bradley-Terry and its two failure modes; generative reward models with non-determinism, the temperature-0 test correction, template/version/order instability and intransitivity with the rock-paper-scissors counter-argument; verbosity bias and the direction reversal against search, with the style and 50/50 annotator biases; preference-data provenance and implicit user signals; RewardBench v2's grouping structure and tie notion with no numeric scores; the same-family preference-model finding and the shared-pathology concession; post-hoc content filters with the Beijing and wrist-tattoo examples; and the RLHF-plus-best-of-N recommendation with the 100-token max-length case.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt` — the critic-reranking result (about 20% to up to 32% at 16x inference, "approximately constant gain" per doubling), the $10,000-per-task viability framing, the Claude Sonnet 4.5 single-instance-versus-reranked chart as the lecturer's characterisation of a vendor's published evaluation, the partner rendered as "Sweden", the outcome-versus-process reward model distinction with unit tests as the supervision signal, and the unit-test-generation discussion with SWTB and TestGen.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt` — the metageneration framing (generate-and-rerank with a reranker as a subroutine), and the search-error versus model-error distinction used in Q22 to separate a ranker defect from a pool change.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` — the generator-side sources of nondeterminism at temperature 0 (GPU reduction order, multi-GPU aggregation and rounding, MoE routing, quantization) used in Q22 to separate a ranker defect from a candidate-pool change, and the tied-token and unknown-empirical-temperature caveats that bound any determinism claim about the generator.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` — the long-tail behaviour of the sampling distribution and the tokenizer/vocabulary argument that makes a diverse candidate pool necessary in the first place, used in Q13 and Q24.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference for the case-study, cheat-sheet and interview-bank formats.
