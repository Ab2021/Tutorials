# Cheat Sheet: Beam, A* and Best-First Search

> `T02` · **Transcript coverage:** primary · [Case study](../01-case-studies/T02-search-decoding.md) · [Blueprint](../03-design-blueprints/T02-search-decoding/HLD.md) · [Interview bank](../02-interview-questions/T02-search-decoding.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Toy FSA comparison | beam **6 steps** vs greedy **4 steps** | `[T]` CMU lecture 5 |
| Dijkstra-style optimal search | **9 steps**, then combinatorial blowup | `[T]` CMU lecture 5 |
| A* on the same problem | **8 steps**, without expanding `S0→S1` | `[T]` CMU lecture 5 |
| Best-first beam search | **10× speedup, identical results** | `[T]` CMU lecture 5 |
| Heuristic illustration | `h=0.5` vs `h=1.0` — changing h changes expansion order | `[T]` CMU lecture 5 |
| Recombination distance matrix | **16×16** KL matrix for beam 16 | `[T]` CMU lecture 5 |
| Typical beam widths | 4–10 for MT; 1 (greedy) is often competitive on LLMs | `[D]` |

---

## The one-table summary

| Method | Guarantee | Cost | Use when |
|---|---|---|---|
| **Greedy** | none | 1× | default; strongest baseline, hard to beat on open-ended generation |
| **Beam search** | none (approximate) | `k×` | tasks with a *correct* target — MT, summarisation, constrained rewriting |
| **Diverse beam** | none | `k×` + diversity penalty | you want k *different* candidates, not k near-duplicates |
| **Stochastic beam** | unbiased samples | `k×` | you need samples but better than i.i.d. — uses the Gumbel-max trick |
| **Best-first beam** | none, but same output as beam | **~10× faster** | you want beam's result without its cost |
| **Dijkstra / uniform-cost** | **optimal** | combinatorial | tiny search spaces only |
| **A\*** | **optimal if `h` is admissible** | combinatorial | you have a real heuristic — and for LMs you usually do not |

---

## Formulas

**Beam search score** with length normalisation:
```
score(y) = log P(y | x) / |y|^α
```
`α` is `length_penalty`. `α = 0` ⇒ no normalisation (favours short). `α = 1` ⇒ per-token average.
**This one parameter changes the ranking more than the beam width does** `[T]`.
*Typical `α ∈ [0.6, 1.0]` is standard practice `[D]` — **the corpus states no range for `α`**, so treat the
interval as a starting point to sweep, not as a sourced figure.*

**A\* priority**: `f(n) = g(n) + h(n)`
- `g(n)` = cost so far (here: negative log-prob of the prefix)
- `h(n)` = estimated cost to go
- **Admissible** `h` never overestimates ⇒ A* is optimal.

**Semiring view** `[T]`: search is a semiring algebra — `⊕` (addition / min) combines
alternatives, `⊗` (multiplication / +) extends paths. Negative log-probs turn products into sums,
so the "shortest path" in `−log p` space is the highest-probability sequence. This is the same
machinery that makes constrained decoding work ([T03](T03-constrained-generation.md)).

**KL-based recombination**: cluster hypotheses whose next-token distributions are within KL
threshold `θ`. Truncation length `n` controls aggressiveness — `n = ∞` is no clustering, `n = 2–3`
is aggressive `[T]`.

---

## Why A* does not work for LLM decoding

`[T]` CMU lecture 5 — the single most important negative result in this topic.

1. **The graph is exponential.** Every token is a branching factor of `|V|` (~128k). You cannot
   enumerate.
2. **No admissible heuristic exists.** Admissibility requires `h` to never *over*estimate the
   remaining cost. But when a model has memorised a passage, `P(next) ≈ 1`, so the true remaining
   cost is ≈0 and any nonzero `h` overestimates — **violating admissibility**. Memorisation
   destroys the guarantee.
3. **Admissibility is sufficient but not necessary.** A constant offset preserves ordering, so in
   practice people use inadmissible heuristics anyway and lose the guarantee knowingly.

**Practical substitute:** a *future-cost* heuristic from a small auxiliary model — train it with an
auxiliary loss or post-hoc, predicting the cost to complete from a partial prefix. This is
structurally the same idea as an RL value function `[T]`. DeepSeek V3 or a small Qwen can serve as
the heuristic model `[T]`.

---

## The curse and blessing of beam search

**Curse** `[T]`: beam search degenerates. Because it optimises likelihood over the whole sequence,
it prefers bland, high-probability continuations — the "likelihood trap". Longer beams make it
*worse*, not better, on open-ended generation. This is why sampling beats beam for chat.

**Blessing** `[T]`: for tasks with a real target, the *local information density* is more uniform —
the standard deviation of surprisal across steps is lower — so the beam's pruning does not
systematically discard the correct path.

**Search error vs model error** `[T]`: these are distinct and you must separate them when
debugging. A search error means the right sequence existed in the beam's reach but was pruned. A
model error means the model assigned it low probability. Only the first is fixable by widening the
beam.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Beam output is bland/generic | curse of beam search | use sampling — beam is the wrong tool |
| Beam output is too short | `length_penalty` too low | raise `α` |
| Beam output rambles | `length_penalty` too high | lower `α` |
| Diverse beam outputs are near-identical | diversity penalty too weak | raise Hamming / n-gram penalty |
| Same result as greedy regardless of beam width | the model is very peaked | expected — memorised/structured output |
| A* runs forever | no admissible heuristic in practice | abandon exact search |

---

## Gotchas

- **Beam width is not a quality dial for chat.** Going 1 → 4 → 10 on open-ended generation
  frequently makes output *worse*. This is the most common misuse `[T]`.
- **Length penalty is usually the bug.** Tune `α` before touching `k`.
- **Diverse beam needs a diversity *metric* choice** — Hamming distance, cumulative diversity, or
  n-gram overlap — and they behave differently. Pick based on whether you care about token-level or
  phrase-level variety `[T]`.
- **Stochastic beam search is the right way to sample.** Naive beam gives you the top-k sequences,
  which is not sampling; the Gumbel-max trick restores unbiasedness while keeping beam's
  efficiency `[T]`.
- **Recombination only helps if the distance metric is right.** Euclidean, cosine and KL over
  next-token distributions give different clusterings; KL is the principled one for probability
  distributions `[T]`.
- **BLEU/ROUGE measure n-gram overlap, not correctness.** They are why beam search looked good for
  years on MT and why it looked bad the moment we measured with LM-as-judge `[T]`.
- **Evaluate diversity separately** — unique-word ratio and bigram overlap. A beam that produces 5
  near-identical candidates wasted 4/5 of your compute `[T]`.

---

## When to use what

| Task | Method |
|---|---|
| Open-ended chat / creative | **sample** ([T01](T01-sampling-decoding.md)) — not beam |
| Translation, summarisation, known target | beam `k=4–6`, `α` tuned |
| Structured extraction | greedy + constrained ([T03](T03-constrained-generation.md)) |
| Reasoning with verification | sample `n` then vote/verify ([T04](T04-test-time-compute.md), [T05](T05-verifiers-best-of-n.md)) |
| You want beam's quality at lower cost | **best-first beam search** (~10× faster) |

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_5_A_and_Best_First_Search.txt`
