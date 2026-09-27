# Interview Bank: Sampling & Decoding Strategies

> `T01` · **Transcript coverage:** primary · [Cheat sheet](../00-cheat-sheets/T01-sampling-decoding.md) · [Case study](../01-case-studies/T01-sampling-decoding.md) · [Design blueprint](../03-design-blueprints/T01-sampling-decoding/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer here is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a lecture statement with the speaker named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here, with assumptions shown. Where the corpus does not supply a figure, the answer says so rather than inventing one.

Questions are ordered to read as one interview: foundations, then mechanism, then the tradeoffs, then debugging, then design at scale.

---

### Foundations

#### T01-Q1 · What is the model actually producing at each decoding step?
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** Strip away the frameworks. At each step of generation, what is the object the model produces, and what does "sampling" mean in relation to it?
**Model answer:** The model is "just a conditional probability distribution" over the vocabulary at each step `[T]` (CMU lecture 3, Amanda). For a context of `n` tokens it defines `P(x_{n+1} | x_1..x_n)` over the whole vocabulary, and generation is the repeated application of that distribution — the chain rule of probability, one factor at a time. Sampling is then any procedure that draws a token from that distribution, and every method in this topic is a *reshape* of the distribution rather than a change to the model. Ancestral sampling draws from it unmodified and is "the property that we're recovering the model distribution exactly" `[T]`. Temperature reshapes it, truncation cuts it, penalties subtract from it, and constrained decoding masks it. The consequence worth stating: if you truncate, you are no longer sampling the model, and any claim about "what the model would say" is a claim about a biased distribution you constructed.
**Signal:** Names the distribution as the primitive and treats every technique as a transform on it — rather than listing sampler names and hoping.
**Follow-ups:**
- *What does "the model" mean once you truncate?* — Point at the truncation-as-deliberate-bias framing; the model distribution is unchanged, your sampling distribution is not.
- *Where does the chain rule come in?* — It is why generation is sequential and why one flipped token changes everything after it.
**Red flags:** Describes sampling as "picking the next word" with no distribution; treats temperature and truncation as the same kind of operation.

#### T01-Q2 · Temperature: the formula, and what its limits mean
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Write the temperature transform and explain what happens at `T = 1`, `T → 0`, and `T → ∞`. Which one is the "correct" setting and why?
**Model answer:** Temperature is `p_i^(1/T) / Σ_j p_j^(1/T)` applied to the probabilities before drawing. At `T = 1` it is "a no-op because you get one over one, this is one, and then E and log cancel out" `[T]` (CMU lecture 2). As `T → ∞` every value goes to 1 and the distribution becomes uniform — you are sampling noise. As `T → 0` you get "a one-hot vector," with the caveat that exactly tied tokens yield a uniform distribution over the tied set `[T]`. The correctness answer is the one that surprises people: **`T = 1` is the only temperature that can see the true probability distribution**, "the only temperature where you can accurately sample from the joint distribution," and everything else "is like a biased distribution that the model isn't actually generating" `[T]`. So any setting other than 1 is a deliberate distortion. That does not make `T = 0.2` wrong — it makes it a decision you have to own, and it is why reasoning models pin temperature to 1 by convention: "the OpenAI reasoning models enforce temperature of one, and you're not allowed to do other temperatures" `[T]`.
**Signal:** States `T = 1` as the only unbiased setting *and* can still justify running lower — the two halves that separate understanding from dogma.
**Follow-ups:**
- *So is temperature 0.2 a bug?* — No: it is a bias you chose; the question is whether the eval supports it.
- *What is the empirical temperature at `T = 0`?* — The lecturer's own open question — "I don't know the answer to that" `[T]`; treat 0 as argmax with unspecified tie-breaking.
**Red flags:** Claims `T = 0` is "no randomness" or "deterministic"; does not know the `1/T` exponent form.

#### T01-Q3 · Why does any truncation exist?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Models put probability on every token in the vocabulary. Why is that a problem, and what is the argument for cutting the tail?
**Model answer:** Because vocabulary is large and the tail is fat. The lecture's slide was updated from Llama's 32,000 tokens to "**Llama 3 has 128k vocabulary tokens**," and the argument is that "if every individual token that is sort of not in the reasonable 100 or 500 tokens to predict next has a tiny amount of probability, these small probabilities add up really really quickly" `[T]` (CMU lecture 3). The visualisation marks the point holding 50% of the mass, so "the tail from that point onwards is half of the probability mass" `[T]`. That is the whole case: half the mass sits outside the head, so an unmodified draw has a substantial chance of picking something the model would not have chosen if it were being careful. The mechanism of harm is a compounding one — generation is autoregressive, so once a tail token enters the context it conditions everything after it. The lecture's diagnostic ladder reflects this: repetitive output means "maybe you're sampling from the long tail," and outright nonsense means "you're definitely sampling from the long tail" `[T]`.
**Signal:** Gives the 50%-of-mass number and connects the tail to autoregressive compounding, rather than saying "low-probability tokens are bad."
**Follow-ups:**
- *What does the tail size do as vocabulary grows?* — It grows; a fixed top-p then keeps more tokens, so truncation behaviour is tokenizer-dependent `[D]`.
- *Is the tail ever the point?* — Yes — the quality dashboard must sample it to measure the model `[T]`.
**Red flags:** Says "we truncate to make output better" with no mechanism; believes top-p removes "bad" tokens rather than reshaping the distribution.

#### T01-Q4 · Top-k and its structural flaw
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** What does top-k do, and what is wrong with a fixed `k`?
**Model answer:** Top-k keeps the `k` most probable tokens and renormalises. The problem is that a fixed count is not a fixed constraint, because the distribution's shape changes from step to step. The lecture's worked example makes this concrete: after "the" the top six tokens hold only **68%** of the mass, while after "the car" the top six hold **99%** `[T]` (CMU lecture 3). The same `k = 6` is therefore a loose cut at one step and a tight one at the next. On a flat distribution `k` is too small — you cut into legitimate alternatives — and on a peaked one it is too large, admitting junk. It is also blind to the absolute probability level, so `k = 50` on a distribution where the top token has 0.9 mass keeps 49 tokens holding 0.1 between them. Top-k survives because it is simple and gives a *hard bound on candidate count*, which matters when a downstream scorer's cost is linear in candidates, but it is the wrong default.
**Signal:** Reaches for 68%-vs-99% unprompted — the canonical number for why a fixed `k` is not a fixed constraint.
**Follow-ups:**
- *When is a hard candidate bound worth the quality loss?* — When a reranker or a constrained decoder's cost is linear in candidates.
- *What replaces it?* — Top-p, then entropy-based methods; see Q5 and Q8.
**Red flags:** "Top-k keeps the k best tokens" and stops; no awareness that the distribution's shape varies by step.

#### T01-Q5 · Top-p, and why it is the default
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Explain nucleus sampling. Why is it usually preferred over top-k, and what is its failure mode?
**Model answer:** Top-p specifies the *mass* rather than the count: seed with the most probable token and "add on more and more tokens into your sample until you've hit that much probability mass," then renormalise `[T]` (CMU lecture 3). Because it adapts to the distribution's shape, the same `p = 0.9` is a tight cut on a peaked step and a loose one on a flat step — precisely the adaptivity top-k lacks, and the reason it is the de facto standard. The failure mode is the long tail: when the distribution is flat, the set reaching 0.9 contains a great many low-probability tokens, and renormalisation hides how much mass was genuinely dropped. The lecture's own framing is that sampling 100–200 tokens per paragraph with a loose top-p makes a tail draw "pretty nasty" `[T]`. It also has no absolute floor — a token with probability 0.001 can survive if it happens to sit inside the cumulative mass — which is the gap epsilon sampling fills.
**Signal:** Explains the adaptivity as the *reason* for the preference, and names the flat-distribution failure rather than claiming top-p is simply better.
**Follow-ups:**
- *What does renormalisation hide?* — The size of the discarded mass; instrument survivors and dropped mass rather than trust the setting.
- *What is the practical fix on flat distributions?* — Add an absolute floor (epsilon) or lower temperature; see Q8.
**Red flags:** Says top-p is "better than top-k" with no mechanism; believes `p = 0.9` guarantees 90% of the mass is retained in a meaningful sense.

#### T01-Q6 · What does `temperature = 0` actually mean?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A product team asks for "temperature 0 so it's deterministic." Is that accurate?
**Model answer:** It is accurate about intent and wrong about fact. `T → 0` yields "a one-hot vector" over the argmax token `[T]` (CMU lecture 2) — so far so good. Three things break the determinism claim. First, ties: the lecture notes that "if two tokens have exactly equal probability you get a uniform distribution over the tied most-probable tokens" `[T]`, so the output is a draw, not a constant. Second, the empirical temperature is not actually zero — asked what it is, the lecturer's answer is "I don't know the answer to that," with a student suggesting roughly 0.01 `[T]`. Third, and decisively for production, the argmax is not stable: floating-point reduction order inside the GPU, MoE routing and quantization all perturb it, and "if the maximum vocabulary item changes just once in your like 1,000 token generated sequence, then it will change everything that happened after that" `[T]`. The honest framing is that `temperature = 0` means **argmax with unspecified tie-breaking**, never "no sampling." Promise reproducibility only conditional on a pinned stack — see Q23.
**Signal:** Separates "the intent of `T = 0`" from "the guarantee `T = 0` delivers," and volunteers the autoregressive amplification.
**Follow-ups:**
- *What would you promise instead?* — Replay of the recorded completion; Q23 gives the full contract.
- *What makes it worse?* — MoE, quantization, and multiple GPUs `[T]`.
**Red flags:** "Temperature 0 is greedy so it's deterministic"; no awareness of reduction-order nondeterminism.

#### T01-Q7 · Greedy decoding: the trap and the legitimate uses
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** What actually goes wrong with greedy decoding, and when is it still the right choice?
**Model answer:** Greedy takes the argmax at every step. Two documented failure modes. **Repetition:** "once you've repeated it twice, you're going to repeat it a third time," and taking the argmax "reinforces this repetition" `[T]` (CMU lecture 4) — the literature example is a ~10–15 token loop in a GPT-3 short story `[T]`. The mechanism is that the repeated context makes the repetition more likely, so the decoder locks in. Base MLE models repeat far more than RL-trained ones, which carry "explicit bias against repeating yourself" `[T]`. **Ordering artefacts:** argmax pushes entity names late — "the dog was seen by Jane" instead of "Jane saw the dog" `[T]` — because the locally-most-probable continuation is not the globally-best construction. Greedy's legitimacy is elsewhere: it is the cheapest path, and it is right for structured extraction, short answers, classification, and anywhere an audited replay matters more than prose quality. The nuance: on natural-language surfaces, prefer *low-temperature sampling* over pure argmax — you keep most of the stability and lose the lock-in.
**Signal:** Distinguishes "deterministic-ish" from "high quality," and knows the repetition mechanism rather than just calling greedy "boring."
**Follow-ups:**
- *Does greedy produce the highest-likelihood output?* — No — that is beam search's property; greedy is myopic, Q15.
- *Why do RL-trained models repeat less?* — Explicit training pressure against repetition `[T]`.
**Red flags:** "Greedy is safe but boring"; recommends greedy universally for reproducibility without noting the argmax instability.

#### T01-Q8 · Entropy-aware sampling: typical, epsilon, ADA, Mirostat
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Walk through the sampling methods that use the distribution's *entropy* rather than a fixed count or mass. What do they buy, and what is the counter-intuitive cost?
**Model answer:** Four methods, all reshaping by information content. **Epsilon sampling** cuts the tail at an absolute probability floor — it gives you a hard quality floor that top-p cannot, because top-p has no notion of an individual token being too unlikely. **Locally typical sampling** sorts the distribution "by closeness to H instead of by absolute value" `[T]` (CMU lecture 3) and then truncates, targeting the typical set. Its counter-intuitive property is load-bearing: it can "cut off things that are relatively high probability," and it produces **higher perplexity** output than ancestral sampling — which is the point, because the always-obvious token is the failure mode it exists to fix. **ADA sampling** combines an epsilon threshold with a distance-from-entropy threshold and is "in practice not often terribly different from… either locally typical or epsilon sampling but it's a little bit faster to compute" `[T]`. **Mirostat** inverts the control entirely: you specify a target perplexity and it uses "a continuously updating update to try to generate tokens that you think will result in that final perplexity being close to that value" `[T]`. Mirostat's catch is that it needs a target per model and per task, and on short outputs the running estimate never converges.
**Signal:** Knows that locally typical sampling deliberately raises perplexity, and can say why that is correct rather than a defect.
**Follow-ups:**
- *Which would you pick for a surface whose failure mode is blandness?* — Locally typical.
- *Why is Mirostat hard to productionise?* — A target perplexity per model and task, and no convergence on short outputs.
**Red flags:** Treats all truncation methods as interchangeable; cannot explain why a method would intentionally increase perplexity.

---

### Mechanism

#### T01-Q9 · Processor ordering: mask, temperature, truncate
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You have a schema mask, a temperature transform and a top-p truncation. In what order do they run, and what breaks if you get it wrong?
**Model answer:** Three orders are possible and two of them are wrong. **Temperature → top-p → mask** reshapes then truncates, and the mask then removes schema-invalid tokens. It fails because truncation can spend the entire probability budget on tokens the schema forbids: at high temperature the survivors can be almost entirely invalid, leaving you sampling from a set far smaller than intended. **Mask after truncation** is the fastest path when the mask is sparse, but it can leave an **empty candidate set** — the classic "the grammar forbids every token the sampler kept" deadlock. **Mask → temperature → top-p** is correct, and the reason is a number: the lecture's JSON-as-FSA observation is that "this is a really narrow constraint at any individual decoding step. We have over 100,000 choices but only like maybe 10 of them that will give us valid JSON at the end" `[T]` (CMU lecture 6). With 10 valid tokens out of 100k, truncating before masking is almost always wrong — you are spending a budget on tokens you will discard anyway. The cost of masking first is that the mask must be cheap enough to run before every truncation; a slow FSM delays every step.
**Signal:** Names the empty-candidate deadlock and the 10-in-100k ratio, and states the cost of the correct order rather than presenting it as free.
**Follow-ups:**
- *What if the mask is genuinely expensive?* — Cache the FSM state per sequence; the transition cost, not the mask cost, is what matters.
- *Does masking interact with token healing?* — Yes — template-driven generation creates "token boundaries that are quote unquote unnatural" `[T]`; see [T03](../01-case-studies/T03-constrained-generation.md).
**Red flags:** Orders by intuition; has not considered that truncation can empty the valid set.

#### T01-Q10 · Repetition penalties: mechanism and collateral damage
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** How do repetition, presence and frequency penalties work, and when do they actively hurt?
**Model answer:** All three subtract from the logits of tokens already in the context, before the softmax. **Presence** applies a flat penalty once a token has appeared; **frequency** scales the penalty with the count; a plain **repetition penalty** divides or subtracts on the logit of any token present in a window. They exist because the repetition trap is real — greedy decoding locking into a ~10–15 token loop `[T]` (CMU lecture 4) — and because the fix must be applied to the distribution, not the output text after the fact. The collateral damage is where candidates separate: **factual recall depends on repetition being correct.** If the answer is a policy number, an entity name, or a code identifier that legitimately recurs, a penalty suppresses exactly the token you need. The second hazard is interaction: penalties are applied before truncation, so raising a penalty reshapes the distribution that top-p is about to cut, which means a penalty change silently changes truncation behaviour. That is why penalties are tuned last and one at a time. Note also that the penalty is a *bias*, like temperature — it moves you further from the model distribution, so it belongs in the same eval gate as any other sampling parameter.
**Signal:** Volunteers the factual-recall counter-case and the penalty/truncation interaction without being prompted.
**Follow-ups:**
- *Where should the penalty sit in the processor chain?* — Before truncation, after masking.
- *What is the alternative to penalties?* — Sampling rather than argmax in the first place, or a different truncation family.
**Red flags:** Calls penalties "always good for long outputs"; does not know they alter the pre-truncation distribution.

#### T01-Q11 · Why truncation is a deliberate bias
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** A colleague says "we truncate to remove the bad tokens." Correct them, carefully.
**Model answer:** Truncation does not know which tokens are bad. It removes tokens by a rule — a count, a cumulative mass, or a distance from entropy — and everything it removes is, by construction, a token the model assigned some probability to. So the output is drawn from a *different* distribution than the model defines, and the honest description is that truncation is a **bias you inject deliberately**. The lecture's statement that `T = 1` is "the only temperature where you can accurately sample from the joint distribution" `[T]` (CMU lecture 2) is the general principle: any reshape takes you off the model's distribution. This matters practically in three places. Evaluation: if you measure a truncated model, you have measured the model *plus your sampler*, and the dashboard surface must therefore run temperature-only with no truncation `[T]`. Reporting: a claim about "what the model would say" is unsupported once you truncate. And diversity: locally typical sampling deliberately produces higher-perplexity output than ancestral `[T]` — a bias that exists precisely to make output less predictable, which is incoherent under the "remove bad tokens" framing.
**Signal:** Reframes truncation as bias rather than filtering, and connects it to why the quality dashboard must run untruncated.
**Follow-ups:**
- *Then why truncate at all?* — Because the unbiased draw includes the tail, and the tail is half the mass; Q3.
- *How do you report a model's quality honestly?* — State the sampler, or measure untruncated.
**Red flags:** Describes truncation as removing "incorrect" or "low-quality" tokens; does not distinguish the model's distribution from the sampling distribution.

#### T01-Q12 · The defaults problem
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** You adopt a new open-weight model. What do you check before setting sampling parameters, and why does the answer matter more than it looks?
**Model answer:** Read the `generation_config` and the library fallbacks, and do not assume either. HuggingFace's fallbacks are `top_p = 0.95` and `top_k = 50`; the model's own config usually overrides them — the lecturer read Llama's as `temperature = 0.6, top_p = 0.9` (the transcript renders the digit as "6"; treat it as an ASR artefact) and Qwen 3 as setting both a top-p and a top-k with `do_sample: true` `[T]` (CMU lecture 3). The operational rule is the core of a policy service: "if you're specifying some generation parameters, it's a good idea to check what the defaults are, check what the model config says they are, and then override everything that you care about" `[T]`. The reason those numbers are not arbitrary is that vendors "done some kind of exhaustive hyperparameter sweep" — "those numbers in those generation configs did not come from [thin air]" `[T]`. So they are a real prior, tuned on the vendor's eval. The correct move is to seed from the model card and perturb with your own eval, rather than start from intuition — and critically, to *override explicitly*, because a model upgrade that changes `generation_config` will silently change your product's behaviour.
**Signal:** States the override-everything rule and the vendor-sweep justification, and connects a config change to a silent production behaviour change.
**Follow-ups:**
- *What happens on a model upgrade if you did not override?* — Silent behaviour change; treat the config as part of the model revision.
- *Where do the numbers live in your system?* — An immutable, versioned policy registry — Q18.
**Red flags:** Uses library defaults without checking; assumes the model card's numbers are arbitrary.

#### T01-Q13 · How do you measure diversity?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You have a surface whose whole value is producing six *different* candidates. How do you evaluate it?
**Model answer:** No single score works, so use a small toolkit. The lecture's set: deterministic metrics are **diversity = the ratio of unique words within one generation** (distinct-n) and **cross-generation diversity = bigram or word overlap between outputs**, plus length as a control; and **LLM-as-judge** for fluency, with the prompt "rate the fluency and coherence of this text on a scale of zero to 10. Uh 10 equals perfect. Only respond with a number" `[T]` (CMU lecture 2). The two axes must be reported together, because the measured result is that they move against each other: greedy decoded text "got a score of 5.3, which I think is not that bad," while diversity was low for greedy and rose with temperature — but "the fluency went down," and "this is a very common issue… as you sample more diverse things, the the quality goes down. And temperature sampling is particularly well known for this" `[T]`. So the artifact is a **Pareto curve**, not a number, with the chosen policy marked as a defensible point on it. The forward pointer in the same lecture is that some methods are "Pareto optimal with respect to diversity and quality" — better on both axes than temperature — which is the case for sampling plus reranking.
**Signal:** Refuses to collapse to one number, names both axes, and produces a Pareto curve with a justified operating point.
**Follow-ups:**
- *What is the judge's role, and its limit?* — Fluency only, never the same family that generated; Q14.
- *How does reranking change the curve?* — It moves the frontier rather than sliding along it.
**Red flags:** Proposes a single "diversity score"; uses fluency alone as the gate for a diversity product.

#### T01-Q14 · LLM-as-judge: what breaks it
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** You want to gate decoding-policy changes on a judge score. What do you have to guard against?
**Model answer:** Three documented problems. **Don't trust it blindly** — "language models are very bad judges for a lot of things and they're particularly bad at things judging things where they themselves are not good at them" `[T]` (CMU lecture 2). **Self-preference** — "Claude will like really like Claude, GPT will really like GPT" `[T]` — so never judge with the same model family that generated; that bias will systematically favour whichever candidate your generator resembles. **Capability creep in the wrong direction:** a judge that is weaker than the generator cannot grade it. The design consequences: use the judge for *fluency and coherence only*, which is the axis where judge scores correlate with human judgement, and keep deterministic metrics (distinct-n, cross-generation overlap, length, JSON validity) as the primary gate, because they are cheap, stable and unfakeable. Sample human review at the tails — the 5th percentile of judged outputs — rather than reviewing the mean, since the mean is where a judge is most reliable and least informative. And re-baseline when the judge model itself changes: a judge upgrade is a silent eval change.
**Signal:** Names self-preference and the domain-competence limit, and specifies using the judge for fluency rather than for the whole verdict.
**Follow-ups:**
- *What replaces the judge for correctness?* — A task-specific verifier or a deterministic metric; see [T05](../01-case-studies/T05-verifiers-best-of-n.md).
- *What happens when the judge model is upgraded?* — Scores shift; treat it as a re-baseline event.
**Red flags:** Uses a judge as the sole quality gate; judges with the same family that generated.

---

### Tradeoffs

#### T01-Q15 · Why beam search is not in production
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Beam search maximises sequence likelihood. Why do frontier products not use it for open-ended generation?
**Model answer:** Two independent reasons that reinforce each other. **The curse of beam search:** on 2018–2020-era models, "as you increase the beam size… the performance downstream actually goes down" `[T]` (CMU lecture 4) — more search makes the output worse on the downstream task. **The likelihood trap:** the highest-probability completions are "less preferred… by humans" than things "still… in the top quarter of a percentile of probability scores" `[T]`, so maximising likelihood walks *away* from the human-preferred region. The deeper point is that the true mode can be degenerate: "the true mode is the empty string" `[T]` — the highest-likelihood sequence under an autoregressive model is frequently trivial or repetitive, and beam search is a machine for finding it. The lecturer's own explanation for why "people don't really use beam search for sort of frontier models anymore" `[T]` is that the likelihood trap is a *model* error, and better models have a human-preference curve that flattens at the top. That is the important nuance: beam search is not permanently wrong, it was wrong for the models of its era. For our estate it stays out of production, but the reasoning is model-conditional, not absolute.
**Signal:** Distinguishes the curse (search pathology) from the likelihood trap (model pathology) and knows the trap may shrink with better models.
**Follow-ups:**
- *So when would you use it?* — Q16, the blessing result.
- *What is the fix if you want high-likelihood output?* — Don't; use sampling plus a verifier — [T05](../01-case-studies/T05-verifiers-best-of-n.md).
**Red flags:** "Beam search is better but too slow" — wrong reason entirely; no awareness of the likelihood trap.

#### T01-Q16 · When beam search is right: the blessing, and diverse beams
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** Given all that, name the cases where a beam is the correct tool — including the variant that exists only to produce *different* candidates.
**Model answer:** Three cases. **A well-defined target with an exact objective** — machine translation into a known target language is the canonical one, where the objective is closer to what beam search optimises. **Tasks where uniform information density is desirable:** the *Blessing of Beam Search* result is that beam search enforces uniform information density, which is why it helps there `[T]` (CMU lecture 4) — the mechanism is that likelihood maximisation spreads information evenly, which is exactly wrong for creative text and exactly right for some structured prose. **Diverse beam search**, when you need `k` genuinely distinct candidates and sampling's diversity is unreliable: the mechanism is to "encourage each new output we decode to be different from everything else that we've decoded up to that point" `[T]`, and it costs `t + g − 1` steps for `g` groups `[T]`. Its costs are honest ones: four penalty terms to tune (Hamming, cumulative, n-gram, embedding similarity), and the embedding-similarity term is "not actually worth the extra computational cost" `[T]`. There is also a fairness flaw worth naming: penalising a token by its prior count is unfair when the token is legitimately correct at a different position `[T]`. On our estate, diverse beam search is the fallback for a `k`-distinct-candidates surface if sampling-plus-reranking cannot hold the diversity floor.
**Signal:** Can argue *for* beam search with a mechanism, and prices diverse beam search's four penalties and `t + g − 1` cost.
**Follow-ups:**
- *Why is uniform information density sometimes good?* — It spreads content evenly, which matches some target distributions better than the human-preference curve.
- *What is the fairness flaw?* — Count-based penalties punish legitimately repeated tokens.
**Red flags:** Cannot name any case where beam search wins; has never heard of diverse beam search or the `t + g − 1` cost.

#### T01-Q17 · Sampling plus reranking versus a bigger model
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You can either generate one candidate from a large model or six from a small one and rerank. How do you choose, and what is the ceiling on the reranking approach?
**Model answer:** The reranking approach is the only lever that moves both axes at once: it gives you diversity *and* quality, and it is the one technique that improved the case where a strong model preferred a weak model's output `[T]` (CMU lecture 4). The lecture's own demo used a **medium model to rerank small-model outputs**, scored by sequence log-probability "normalized by length… that makes it easier to compare longer and shorter things on the same footing" `[T]` (CMU lecture 2) — length normalisation is essential or the scorer just prefers longer text. Cost is `n`× generation plus a cheap scoring pass, which is usually far below the cost of one large-model generation, because generation is the expensive term and scoring is a single parallel pass over completed sequences. **The ceiling is the scorer.** If the scorer is weaker than the generator, reranking cannot recover the gap — this is the KL bound in [T05](../01-case-studies/T05-verifiers-best-of-n.md), which quantifies how much of the target distribution best-of-n can reach. Two practical caveats: the scorer must be *better at ranking than generating*, which is not automatic, and a sequence-level log-probability scorer is not a preference model, so for subjective quality you need a trained reward model.
**Signal:** States the scorer-quality ceiling and the length-normalisation requirement, and knows when a log-prob scorer is insufficient.
**Follow-ups:**
- *What is the cost ratio?* — `n`× generation plus one scoring pass; generation dominates — see [T05](../01-case-studies/T05-verifiers-best-of-n.md) for the asymmetry.
- *When is a log-prob scorer enough?* — When correctness, not preference, is what you are ranking on.
**Red flags:** Claims reranking "always beats a bigger model"; no awareness that the scorer bounds the gain.

#### T01-Q18 · Where should the sampling policy live?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Six product surfaces share one model on one GPU pool and none of them wants the same decoding behaviour. Where do you put the configuration?
**Model answer:** Four options with a clear winner at scale. **Application code** setting parameters per call is fast for one team and breaks the moment a second consumer shares the model: no audit trail, parameters drift per PR, and "temperature 0.2" means different things on different surfaces. **The model's `generation_config` alone** gives vendor-tuned defaults for zero effort, and is exactly one config for six surfaces — a baseline during bring-up, never a final answer, because a model upgrade silently changes behaviour. **A platform policy service** — a pure function from `(surface, model_id, policy_version)` to an immutable policy record — is the right answer for a shared pool: replay becomes a request plus a `policy_id` rather than a request plus six floats someone may have edited, the eval gate becomes a release gate, and cost modelling is possible because the parameter space is closed. Its real cost is organisational: the central team becomes a bottleneck, so it needs a fast exception path or teams will route around it — hence an explicit `research` escape hatch. **Per-request switching driven by a difficulty router** matches decoding to difficulty and the lecture calls it "a really reasonable thing to do" `[T]`, but no published system does it `[T]`, it doubles the (policy, request) evaluation surface, and regressions become unattributable. Defer it until the policy service is stable.
**Signal:** Chooses the policy service and argues the *organisational* cost, not just the technical benefits — plus the escape hatch that keeps it from being bypassed.
**Follow-ups:**
- *What is the strongest argument against centralising?* — The bottleneck; a legitimate one-off need with no fast path will route around the service.
- *What makes it auditable?* — `policy_id` on every response, immutable versions, resolver as sole writer.
**Red flags:** Says "put it in a config file"; centralises without acknowledging the bottleneck; no versioning story.

#### T01-Q19 · Adaptive computation for decoding: what does the corpus actually support?
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** You want to spend more compute on hard requests and less on easy ones, at the decoding layer. What is real, what is a research direction, and what would you measure?
**Model answer:** Three tiers. **Real and deployed:** routing by difficulty to different *models*, which the LLMOps cost talk calls the "single highest-leverage pattern" and reports cutting total spend by half with no quality drop users notice `[T]` — that is a model-routing decision, not a decoding one. **Real with a stated mechanism but no published savings:** adaptive self-consistency, which stops sampling early against a **0.95** confidence threshold on a Beta posterior over the top-1 and top-2 outputs `[T]` (CMU lecture 7). The corpus gives the threshold and states that fixed self-consistency costs "100 times the inference cost" for 100 samples `[T]`, but **gives no figure for samples saved** — so we instrument it rather than claim one. A worked expectation with an *assumed* mean of 5 samples to reach 0.95 versus a fixed 20 gives `(20 − 5)/20 = 75%` reduction, and the 5 is an assumption, labelled `[D]`. **Research:** per-request *decoding-parameter* switching driven by a difficulty router — the lecture calls it "a really reasonable thing to do" `[T]` but no published system does it `[T]`. What I would measure: samples-to-threshold distribution, the accuracy of the early-stopped answer against the full-sample answer, and the cost saved net of the router's own inference.
**Signal:** Separates deployed from mechanism-only from research, and explicitly refuses to quote a savings figure the corpus does not contain.
**Follow-ups:**
- *Why is the 0.95 threshold on top-1/top-2?* — It is a cheap posterior proxy; more candidates make the stopping rule expensive.
- *Where does the router cost show up?* — It is inference too; net savings must subtract it.
**Red flags:** Quotes a self-consistency saving figure from nowhere; conflates model routing with decoding-parameter routing.

#### T01-Q20 · The diversity/quality frontier and Pareto methods
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A product wants maximum diversity at acceptable fluency. Temperature gives you the tradeoff but moves you along one curve. Is there anything that moves the curve itself?
**Model answer:** Yes, and naming it is the senior move. Temperature sampling is famously on the wrong side of this tradeoff — "as you sample more diverse things, the the quality goes down. And temperature sampling is particularly well known for this" `[T]` (CMU lecture 2) — and the measured illustration is that greedy scores 5.3 on judge fluency with low diversity, while raising temperature raises diversity and lowers fluency. So temperature trades along a fixed frontier. The lecture's forward pointer is that some methods are "Pareto optimal with respect to diversity and quality" — better on *both* axes than temperature `[T]`. The practical member of that family is **sampling plus reranking**: generate `n` candidates diversely, then use a scorer to select, which converts the diversity axis into a candidate pool and the quality axis into a ranking, so you get both. The other candidate is **locally typical sampling**, which targets the typical set rather than the head and so avoids both the always-obvious token and the tail draw. The engineering discipline that follows: publish the **Pareto curve** per policy version rather than collapsing to one number, and mark the chosen operating point with its justification — because there is no single correct setting, only a defensible point.
**Signal:** Distinguishes moving *along* the frontier from moving *the* frontier, and names reranking as the practical Pareto method.
**Follow-ups:**
- *Why not just pick the eval's argmax point?* — Overfitting to the eval; users will not replicate your settings `[T]`.
- *What replaces the curve at 10x scale?* — An automated gate plus sampled human review of the tails.
**Red flags:** Treats temperature as the only diversity lever; reports one number for a two-axis property.

---

### Debugging

#### T01-Q21 · Repetitive output: diagnose it
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A surface starts producing loops — the same 12 tokens over and over. Walk me through the diagnosis.
**Model answer:** Work from the cheapest check outward. First, confirm the sampling policy has not changed: a repetition penalty removed in a refactor, or a surface accidentally moved to pure argmax, produces exactly this and is the most common cause. The mechanism to hold in mind is that the argmax "reinforces this repetition" — "once you've repeated it twice, you're going to repeat it a third time" `[T]` (CMU lecture 4). Second, the lecture's diagnostic ladder locates it by symptom: repetitive output means "maybe you're sampling from the long tail"; outright nonsense means "you're definitely sampling from the long tail" `[T]` (CMU lecture 3) — so repetition points at the tail, and the response is to *tighten* truncation (lower top-p, add an epsilon floor), which is counter-intuitive but correct. Third, check the tokenizer: a recent tokenizer or template change alters what counts as a repeated token and can defeat a working penalty. Fourth, check the model revision — base MLE models repeat far more than RL-trained ones, which carry "explicit bias against repeating yourself" `[T]`, so a swap to a differently-trained checkpoint changes this property. Finally, note the floor: repetition loops around 10–15 tokens are documented under greedy `[T]`, so if you are seeing much longer loops, truncation is not the whole story.
**Signal:** Goes policy-first, then invokes the long-tail ladder to decide the direction of the fix, and names the tokenizer as a non-obvious suspect.
**Follow-ups:**
- *Why is tightening truncation the fix for a tail problem?* — The tail draw is the cause; Q3.
- *What if tightening hurts diversity on that surface?* — Then the surface needs reranking, not a looser sampler.
**Red flags:** "Increase the repetition penalty" as the whole answer; no awareness that the fix direction is counter-intuitive.

#### T01-Q22 · JSON validity fell after a sampling change
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A structured-output endpoint drops from 99.9% valid JSON to about 98% after a policy update. What do you check, in order?
**Model answer:** Four checks, ordered by likelihood. **Processor order first:** if the mask now runs after truncation, or after temperature, truncation can consume the budget on schema-invalid tokens and leave a tiny or empty valid set — the fix is mask → temperature → top-p. The justification is the ratio: at a JSON step there are "over 100,000 choices but only like maybe 10 of them that will give us valid JSON at the end" `[T]` (CMU lecture 6). **Temperature second:** raising temperature flattens the distribution, which both widens the truncation set and makes the mask's survivors more varied — a temperature change is a schema-validity change even though it looks unrelated. **Token healing third:** if the change touched a heuristic that repairs "token boundaries that are quote unquote unnatural" `[T]` created by template-driven generation, validity falls in exactly this pattern — valid-looking output that the parser rejects. **Fourth, the sampler's survivor diagnostics:** an empty-candidate or near-empty event means the interaction between truncation and mask, not the schema. Note the general lesson: a 0.1% → 2% move is small enough to look like noise, which is why JSON validity has to be a *release gate* with an alert, not a dashboard.
**Signal:** Puts processor order first and explains *why* temperature matters to schema validity — a non-obvious causal link.
**Follow-ups:**
- *Why is mask-after-truncate ever chosen?* — It is faster when the mask is sparse, at the cost of the empty-set risk.
- *What is the alert threshold?* — A hard floor per surface, e.g. ≥ 99.9%, gated on release.
**Red flags:** Blames the model; treats 98% as acceptable; does not know the processor-order dependency.

#### T01-Q23 · The reproducibility contract
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Compliance wants a written guarantee that an audited extraction can be reproduced exactly. What do you actually sign?
**Model answer:** Not the naive promise. `temperature = 0` is not reproducible in production, for three compounding reasons: reduction order inside the GPU — threads "all running at the same time," so sums complete in nondeterministic order, with the symptom that "you'll get it the same like five times, but then you'll get a different one" `[T]`; MoE routing, which adds "many many more argmaxes" per step `[T]`; and quantization, where "you'll have more rounding errors" `[T]`, with multi-GPU compounding it because "one GPU is more likely to have a larger difference than the threads within the GPU" `[T]` (CMU lecture 2). The amplification is what makes it fatal: "if the maximum vocabulary item changes just once in your like 1,000 token generated sequence, then it will change everything that happened after that" `[T]`. So the contract has two tiers. **Default: replay of the recorded completion**, bit-exact by construction, zero inference cost, with `policy_id` and the model revision recorded — this is what an audit actually needs, and it does not re-derive the answer. **Escalation: a conditional generation guarantee** — same model revision, same engine version, same GPU model, fixed seed, batch-invariant kernels, continuous batching disabled — which is achievable and testable but costs throughput. The price is real: the cost ladder puts continuous batching as the rung taking cost from 100 to 42 units `[T]`, so turning it off is roughly **2.4x** on that surface `[D]`. Never promise bit-exact regeneration from a live serving config.
**Signal:** Offers the two-tier contract with the escape hatch priced, and cannot be pushed into signing the naive promise.
**Follow-ups:**
- *What does the recording need to contain?* — Completion, `policy_id`, model revision, and enough stack identity to detect a change.
- *When is the expensive tier actually required?* — Only when the audit legally requires a fresh generation.
**Red flags:** Promises bit-exact `temperature = 0`; does not know the three nondeterminism sources; proposes a fix with no cost attached.

#### T01-Q24 · Quality cliff after a model swap
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A model revision ships and judged quality drops on one surface while everything else looks fine. Nothing about the sampling policy changed. What happened?
**Model answer:** The distribution changed shape, so the sampler that was tuned to the old one is now mis-set — and only the surface whose policy sat nearest a boundary notices. Two mechanisms from the corpus. First, "larger models behave very differently" `[T]`, and the same is true across revisions of the same size: the head's mass distribution moves, so a fixed top-p that was cutting at 0.9 mass now keeps a different number of tokens. Second, the model-error versus search-error distinction: "your search errors will be the same for the same decoding method, but your model errors will change radically" `[T]` `[T]` (CMU lecture 3) — so a model swap changes the error you cannot fix with decoding, and the decoding change you make in response only touches search error. The diagnosis to run: pre-truncation entropy histogram and survivor count per step, compared across the two revisions on a fixed prompt set. A shifted entropy histogram proves the distribution moved. The fix is to re-tune the policy from the new model card's defaults rather than patching the old policy, and to **key policy versions to the model revision** so that a model swap forces an eval rather than silently inheriting a stale policy. Also check `generation_config`: the new revision may ship different vendor-swept defaults, which the resolver must still override explicitly.
**Signal:** Diagnoses a distribution-shape change with a measurable instrument (entropy histogram, survivor count) rather than guessing at parameters.
**Follow-ups:**
- *What is the structural fix?* — Policy versions keyed to model revision; a model swap is an eval event.
- *Why does only one surface notice?* — Its policy sat nearest a boundary; the others' settings were not sensitive to the shift.
**Red flags:** Re-tunes blindly; does not know that model error and search error are separable.

---

### Scale and design

#### T01-Q25 · Sampling under multi-tenancy
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Six surfaces, one GPU pool. One surface generates six 600-token candidates per request. What breaks, and is it a decoding problem?
**Model answer:** It looks like a decoding problem and it is a **KV-capacity** problem reached through a decoding decision. The chain: a decoding policy determines output length, output length determines how much KV each request holds, KV occupancy determines how many requests fit in a batch, and batching determines throughput and inter-token latency for everyone. Six 600-token candidates is 3,600 output tokens per brief against roughly 250 for chat — at an assumed 1,800 tokens/s aggregate that is ~2 s of dedicated pool time per brief `[D]`, and critically the six candidates run *concurrently*, so they are six long-lived KV allocations, not one. On a shared pool they evict interactive requests' KV, and the interactive surfaces see a latency spike with no change to their own configuration. So the remedies split by layer: at the decoding layer, cap `max_tokens` per candidate and consider staggering the candidates; at the scheduling layer, isolate by output-length class — the decision that inverts at scale is "one pool for everything." Note the ordering principle: `max_tokens` comes first in tuning because it is the only parameter that bounds worst-case cost and pool occupancy; every other sampling parameter is second-order against it.
**Signal:** Traces the causality from a sampling decision to a shared-resource effect, and names `max_tokens` as the first-order lever.
**Follow-ups:**
- *Why is the effect invisible in the offending surface's own metrics?* — It shows up as other surfaces' latency; see [T08](../01-case-studies/T08-batching-scheduling.md).
- *What changes at 10x?* — Pool splitting stops being optional.
**Red flags:** Treats it as a sampling-quality question; suggests raising top-p or other quality knobs.

#### T01-Q26 · 10x traffic: what actually breaks
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Your estate grows 10x in traffic. Which parts of the decoding design survive, which break, and what inverts?
**Model answer:** **The sampler is the last thing to break.** The truncation sort is a 128k-vector sort, microseconds against a decode step dominated by reading gigabytes of weights, which is why the requirement is "< 2% of a decode step" and why per-request parameter switching is feasible at all. Sampler cost is O(1) in traffic. **The policy service also survives** — its cost is O(number of policies), not O(tokens) — but its *eval gate* becomes the bottleneck, because you cannot hand-run a 200-prompt eval per release across a large surface count. The fix is to automate the gate and sample human review at the 5th percentile rather than the mean; the LLMOps evals guidance is that 200 curated prompts beat 10,000 random ones `[T]`. **The first thing that breaks is the shared pool** — Copy Studio's `n`-candidate generation competing with interactive surfaces for KV — and that forces pool splitting by output-length class, which is a decoding-policy consequence rather than a capacity decision. **What inverts:** per-request difficulty routing stops being optional, because a single default per surface wastes compute on easy requests; and temperature-1 evaluation stops being affordable on demand, so the quality dashboard moves to a scheduled overnight job on a separate pool. **What gets harder, not easier:** determinism, as GPU count, MoE and quantization variants multiply — so the reproducibility contract narrows and completion replay becomes the only audit mechanism.
**Signal:** Correctly identifies the sampler as *not* the bottleneck and the eval gate as the real scaling constraint — the non-obvious answer.
**Follow-ups:**
- *Why does temperature-1 evaluation have to move?* — It is the most expensive per request in the estate and cannot be truncated.
- *Is the policy service worth its cost at 10x?* — Yes — O(policies), not O(tokens); the gate is the cost.
**Red flags:** Says "just scale the GPUs"; assumes sampling becomes a bottleneck; no view on the eval gate.

#### T01-Q27 · Designing a decoding policy for a new surface
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** A new product surface arrives — an agentic coding assistant with 40+ model calls per session, where runs are diffed against each other. Design its decoding policy from scratch and tell me what you would not do.
**Model answer:** Start from the surface's requirement, which is not "good text" but **run-to-run comparability**: the user diffs two runs, so output must be stable enough that a difference means a real difference. That points at near-greedy decoding, and then immediately at the trap — pure argmax falls into repetition loops around 10–15 tokens `[T]` and pushes entity names late ("the dog was seen by Jane") `[T]`, which is worse in code than in prose because identifier ordering carries meaning. So the policy is low-temperature sampling rather than argmax — the lecture's guidance scale puts 0.5 at "a more conservative variety of sampling" against 0 at "greedy sampling… it should always give you the same result" `[T]`. Truncation: a tight top-p with an epsilon floor, because agent output is long and the tail argument compounds across 40+ calls. Mask before truncate wherever the tool calls are schema-constrained. **What I would not do:** promise bit-exact reproduction of a session — the corpus's own experience is that reproducing a coding agent's results ran into exactly this, "we would never be able to get the same results… even on the same instance with like greedy sampling" `[T]`. I would not use beam search, because the likelihood trap bites hardest on long-form generation. And I would not let the session's context grow unbounded, because a policy that lengthens output silently changes KV pressure for every co-tenant — see [T07](../01-case-studies/T07-kv-cache.md).
**Signal:** Derives the policy from the surface's real requirement, and refuses the determinism promise on the strength of a documented failure rather than caution.
**Follow-ups:**
- *Why not argmax for diffability?* — Repetition lock-in and ordering artefacts; low-temperature sampling gets most of the stability without them.
- *What is the agentic-specific hazard?* — Error compounding across 40+ calls, and unbounded context growth.
**Red flags:** Reaches for greedy because "deterministic"; makes a determinism promise for an agentic session; ignores context growth.

#### T01-Q28 · The one number that splits the estate
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** You have to explain to a mixed audience of researchers and product owners why the same estate runs different sampling settings. What is the single organising idea?
**Model answer:** **`T = 1` is the only unbiased temperature, and that splits the estate into measurement surfaces and product surfaces.** The research position is the correct one on its own terms: temperature 1 is "the only temperature that can see the true probability distribution… the only temperature where you can accurately sample from the joint distribution," and at anything else "we're getting something that's like a biased distribution that the model isn't actually generating" `[T]` (CMU lecture 2). So any surface whose job is to *measure the model* must run temperature 1 with no truncation — truncation is a bias you inject, and you cannot measure a model through your own bias. Reasoning-model conventions point the same way: "the OpenAI reasoning models enforce temperature of one, and you're not allowed to do other temperatures" `[T]`. Every other surface's job is to satisfy a *product* requirement, and product requirements are not model fidelity — they are repeatability, diversity, schema validity, or faithfulness. Once you frame it that way, the six surfaces are not six opinions about temperature; they are two classes with a principled dividing line, and each product surface's setting is justified by an eval on its own axis rather than by an argument about which temperature is "right." That framing is what makes the policy service politically survivable: research gets its fidelity, products get their settings, and both are measured.
**Signal:** Produces a principled two-class split rather than a compromise, and uses it to resolve an organisational conflict.
**Follow-ups:**
- *What if a product surface wants truncation for the wrong reason?* — The eval decides; the argument is about the surface's axis, not about fidelity.
- *Does the split survive a reasoning-model migration?* — Yes — it removes the surface from the policy matrix entirely, since temperature is pinned to 1.
**Red flags:** Averages the positions ("use temperature 0.5 everywhere"); argues from preference rather than from what each surface is for.

---

## Whiteboard exercises

### Exercise 1 — Diagnose a decoding regression in 20 minutes
**Prompt.** `Support Copilot` had 99.7% valid JSON and a judged fluency of 8.1 last week. This week it is 97.9% and 8.4, and the distinct-n metric has risen 12%. No deploy is recorded. You have 20 minutes and access to logs, the policy registry, and a canary environment. Produce the diagnosis and the fix.

**What to produce.** A ranked hypothesis list, the specific measurement that confirms or kills each, the processor-chain diagram as you believe it *should* be, and the fix with its rollback.

**Expected whiteboard.**

```
Hypotheses (ranked by likelihood x cheapness to test)
  1. Entropy shift from a model/revision change ........ test: pre-truncation entropy histogram, this wk vs last
  2. Policy drift (version pointer moved) ............. test: policy_id distribution in response logs
  3. Processor order regressed ........................ test: mask-before-truncate assertion + empty-set counter
  4. Tokenizer/template change ........................ test: token counts per prompt before/after

Direction of evidence:
  distinct-n UP + fluency UP + JSON DOWN
        -> more diverse sampling, i.e. a FLATTER distribution
        -> consistent with (1) or (2), NOT with a mask bug
           (a mask bug lowers validity without raising diversity)

Fix path:  pin model revision -> re-run eval gate -> re-tune from model card
Rollback:  flip the resolver's surface->version pointer
```

**Grading rubric.**
- Reads the *direction* of the three metrics jointly and eliminates the mask hypothesis — full marks require the argument that a mask bug would not raise diversity and fluency.
- Names the entropy histogram as the decisive instrument, not just "check the config."
- Gives a fix that is a version pin plus a re-eval, and a rollback that is a pointer flip.
- Does not propose changing a sampling parameter before the cause is identified.

### Exercise 2 — Design the reproducibility contract
**Prompt.** A regulator requires that any of 500,000 audited extractions per month can be reproduced on demand. You may spend at most 10% extra on that surface's infrastructure. Write the contract.

**What to produce.** The guarantee in one sentence, the mechanism, the stack the mechanism pins, the cost arithmetic, and the explicit list of what is *not* guaranteed.

**Expected whiteboard.**

```
Guarantee:  replay of the recorded completion, bit-exact, indexed by request_id
            (NOT: regeneration reproduces the answer)

Recorded per response:  completion | policy_id | model revision | engine version | GPU model | seed

Tier 1 (default, 0% extra):  replay the stored bytes
Tier 2 (escalation, ~2.4x on that surface):  fresh generation, pinned stack
        requires batch-invariant kernels + continuous batching OFF

Cost arithmetic [D]:
  cost ladder puts continuous batching as the rung 100 -> 42 units
  turning it off  = 100/42 = 2.4x on that surface
  10% budget -> Tier 2 can apply to at most ~7% of the volume
                 (0.07 x 1.4 + 0.93 x 0.0 = 9.8%)

NOT guaranteed:  cross-hardware, cross-driver, quantized-vs-unquantized,
                 batched vs unbatched, multi-GPU
```

**Grading rubric.**
- Separates replay-of-record from regeneration and states plainly that regeneration is not guaranteed by default.
- Prices Tier 2 at roughly 2.4x using the 100→42 continuous-batching rung, and shows the budget arithmetic rather than asserting it.
- Lists four or more explicit non-guarantees, including at least one hardware and one batching condition.
- Names the recorded fields, including `policy_id` and the model revision.

### Exercise 3 — Choose a truncation strategy for a three-surface estate
**Prompt.** Three surfaces: (a) an audited extractor that must be schema-valid and replayable, (b) a chat copilot where blandness is the complaint, (c) an offline model-quality dashboard. Pick a truncation strategy for each, justify it against at least one alternative you rejected, and say what you would measure to know you were wrong.

**What to produce.** A decision table with the rejected alternative and the failure signal per surface, plus the processor order for surface (a).

**Expected whiteboard.**

| Surface | Choose | Reject | Why | Wrong-if signal |
|---|---|---|---|---|
| (a) Extractor | mask → temperature 0/low → tight top-p + epsilon floor | top-k | need a hard validity floor and an absolute probability floor | JSON validity < 99.9% |
| (b) Copilot | locally typical sampling | plain top-p | targets the typical set; output is "neither always-obvious nor always-surprising" — and it *deliberately* raises perplexity | judged fluency falls without diversity rising |
| (c) Dashboard | temperature only, NO truncation | any top-p | must measure the model, not a truncated version of it | entropy histogram diverges from the untruncated baseline |

```
Surface (a) processor order:
   [ schema mask ] -> [ penalties ] -> [ temperature ] -> [ top-p + epsilon ] -> draw
        ^ first: 10 valid tokens out of 100k means truncating first wastes the budget
```

**Grading rubric.**
- Rejects top-k for the extractor on the "fixed `k` is not a fixed constraint" argument, citing the 68%-vs-99% drift.
- Chooses locally typical for the chat surface and knows it intentionally produces higher-perplexity output than ancestral.
- Refuses truncation entirely for the dashboard, on the grounds that truncation is injected bias.
- Puts the mask first and justifies it with the ~10-valid-tokens-in-100k ratio; names a measurable wrong-if signal for each surface.

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` — temperature formula and limits, tied-token behaviour and the unknown empirical temperature, GPU non-determinism (reduction order, MoE routing, quantization, multi-GPU) and the autoregressive amplification, LLM-as-judge with the greedy 5.3 score and the self-preference caveat, length-normalised re-ranking, and the diversity/quality tradeoff statement.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt` — the model-as-distribution framing, temperature 1 as the only unbiased setting, the long-tail argument with the 128k vocabulary and the 50%-of-mass point, the 68%/99% top-6 example, HuggingFace and Llama defaults with the override-everything advice, epsilon/locally-typical/ADA/Mirostat, and the long-tail diagnostic ladder.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt` — the repetition trap and the 10–15 token loop, the curse of beam search, the likelihood trap and the empty-string mode, the blessing of beam search and uniform information density, and diverse beam search with its four penalties, the embedding-similarity verdict and the `t + g − 1` cost.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt` — the logit-mask mechanism, the JSON-as-FSA narrow-constraint observation, and token healing.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt` — self-consistency at 100x cost and adaptive self-consistency's 0.95 Beta-posterior threshold, with no samples-saved figure given.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the difficulty-routing pattern, the 100→42→26→11 cost ladder used for the reproducibility arithmetic, and the 200-curated-prompts eval guidance.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
