# IQ-05 — Interview Questions: RNN/LSTM → Attention

| Field | Value |
|---|---|
| **Module** | CS-05 — Why Fine-Tuning Was Hard Pre-Transformer: RNN/LSTM → Attention |
| **Pairs with** | CS-05 (case study), CH-05 (cheat sheet), CS-01 (foundations), CS-13 §6.8 (LoRA) |
| **Total questions** | 98 (30 L1 + 28 L2 + 22 L3 + 8 L4 + 10 L5) |
| **Levels covered** | Screen / Intermediate / Advanced / System Design / Debug |
| **Source videos** | 06 (`Why Finetuning Was Difficult in RNN or LSTM`), 07 (`LSTM vs Transformer`) |

---

## How To Use This File

- **L1** = phone screen / recruiter filter — you get 30 seconds, give a definition plus one number.
- **L2** = working engineer — 2–3 minutes, expects implementation detail and a formula.
- **L3** = senior / specialist — 5 minutes, expects internals, derivations and trade-offs.
- **L4** = staff / system design — 15-minute whiteboard, answer in the order *requirements → constraints → design → trade-offs → failure modes*.
- **L5** = debugging & incident — state your checks **in order**, and say what each check rules in or out.

Every question has **Answer**, **Why the interviewer asks this**, and **Trap** (the plausible-but-wrong answer that gets candidates rejected). `[Company style: ...]` tags mark loops where the question is typical.

The single most important habit for this topic: **do not answer "vanishing gradients" for everything.** Three separate things break long-range dependencies (gradient, information, optimization — see L3 Q5), and interviewers who know the field are listening for whether you can tell them apart.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What does BPTT stand for, and why does it matter?**
- **Answer:** Backpropagation Through Time. You unroll the recurrent network across its `T` time steps to form a feedforward graph, then run ordinary backprop on that unrolled graph. It matters because the unrolling is what makes the loss at step `t` depend on the parameters through *every* earlier step, so the gradient is a sum of products of Jacobians rather than a single local derivative. That product is the source of both vanishing and exploding gradients.
- **Why the interviewer asks this:** The word "through time" is the whole topic. A candidate who says "it's just backprop" has not understood why an RNN is different from an MLP.
- **Trap:** Saying BPTT is a different algorithm from backprop. It is not — it is backprop applied to an unrolled graph. The only difference is memory: you must keep every intermediate `h_t` alive.

**Q2. State the vanishing gradient problem for RNNs in one sentence.**
- **Answer:** The gradient of a loss at time `t` with respect to a state at time `k` is a product of `t − k` Jacobians, `∂h_t/∂h_k = Π_{j=k+1}^{t} diag(1 − h_j²) W_hhᵀ`; when the spectral norm of that product is below 1, the contribution shrinks geometrically with distance until it underflows relative to the local terms in the same sum.
- **Why the interviewer asks this:** Tests whether you know it is a *product*, not a sum. Products are what make it exponential.
- **Trap:** "The gradient becomes zero." It does not become exactly zero — it becomes negligible *relative to the other terms in the same gradient sum*. That distinction is exactly why Adam does not save you (L2 Q6).

**Q3. Why is a *product* of Jacobians so much worse than a sum?**
- **Answer:** A sum of `T` terms grows at worst linearly in `T`. A product of `T` terms with per-term magnitude `λ` grows as `λ^T` — geometric. At `λ = 0.9` and `T = 50`, `λ^50 ≈ 0.0052`; at `λ = 0.5` it is `8.9 × 10⁻¹⁶`, i.e. below fp32 epsilon `1.19 × 10⁻⁷`. Nothing in the architecture is broken; the arithmetic simply runs out of exponent.
- **Why the interviewer asks this:** It separates people who memorised the phrase from people who can reason about magnitudes.
- **Trap:** Quoting the number without checking it. Interviewers frequently ask you to compute `0.9^50` live.

**Q4. Name the four gates of an LSTM and what each does.**
- **Answer:** `f_t` forget gate — decides what fraction of the previous cell state to keep; `i_t` input gate — decides what fraction of the candidate to write; `g_t` (sometimes `C̃_t`) candidate — the new content proposed by the current input; `o_t` output gate — decides what fraction of `tanh(C_t)` becomes the hidden state `h_t`. All four are sigmoids (except `g`, which is `tanh`), each computed from `[h_{t−1}; x_t]` with its own weight matrix.
- **Why the interviewer asks this:** Pure vocabulary check, but the follow-up ("which one is tanh and why") is not.
- **Trap:** Saying the cell state *is* the hidden state. They are different tensors: `C_t` is the internal highway, `h_t = o_t ⊙ tanh(C_t)` is what the next layer sees.

**Q5. Why is the LSTM cell state called a "gradient highway"?**
- **Answer:** Because `C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t`, the partial derivative `∂C_t/∂C_{t−1} = diag(f_t)` — a diagonal matrix of forget-gate values, with no weight matrix multiplied in. So the path from `C_T` back to `C_k` is `Π diag(f_j)`, a product of numbers in `(0,1)` chosen by the network, not a product of weight matrices that can have spectral norm far from 1.
- **Why the interviewer asks this:** This is the single derivation that separates "I read a blog post" from "I understand the architecture".
- **Trap:** Claiming the product is bounded away from zero. It is bounded *above* by 1, but the network can drive `f_t → 0` — which is precisely what it does when it wants to forget. See L2 Q8.

**Q6. GRU vs LSTM — how many gates, and which ones?**
- **Answer:** GRU has two gates (`z_t` update, `r_t` reset) and one state `h_t`, versus LSTM's three gates plus a candidate and two states. GRU merges the LSTM's forget and input gates into a single update gate via `h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t`, so the "keep" and "write" decisions are forced to sum to 1.
- **Why the interviewer asks this:** Practical model-choice question; also tests whether you know GRU is *not* simply "LSTM with fewer parameters = worse".
- **Trap:** Saying GRU "has no cell state so it cannot do long-range". It has an equivalent linear path through `h_t`, and on many sequence-labeling tasks GRU matches or beats LSTM at 75 % of the parameters.

**Q7. What is attention, mechanically, in one sentence?**
- **Answer:** For each query `q_i` you compute a similarity against every key `k_j`, normalise those scores with a softmax into a convex combination, and return the weighted average of the values: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`. Every position reads from every other position in a single matrix multiply — no recurrence, so no gradient product.
- **Why the interviewer asks this:** The one-line version is the gate; the follow-ups probe `√d_k`, masking, and complexity.
- **Trap:** "Attention is a lookup table." It is a *soft* dictionary — a convex mixture over all positions. Hard lookup is what you get after argmax, which is not what the layer computes.

**Q8. What are Q, K and V?**
- **Answer:** Three learned linear projections of the same input `X ∈ ℝ^{n×d}`: `Q = XW_Q`, `K = XW_K`, `V = XW_V` with `W_Q, W_K ∈ ℝ^{d×d_k}` and `W_V ∈ ℝ^{d×d_v}`. Q is "what this position is looking for", K is "what this position advertises", V is "what this position will hand over if selected". In self-attention all three come from the same sequence; in cross-attention Q comes from the decoder and K, V from the encoder.
- **Why the interviewer asks this:** Vocabulary, plus it lets them ask about the parameter count of a layer next.
- **Trap:** Forgetting that Q, K, V are *separate* weight matrices. If you said "Q, K, V are the same tensor split three ways" you have confused attention with multi-head splitting.

**Q9. Why divide by √d_k?**
- **Answer:** If the components of `q` and `k` are independent with zero mean and unit variance, then `q · k = Σ_{i=1}^{d_k} q_i k_i` has variance `d_k` and standard deviation `√d_k`. Without scaling, the logits grow like `√d_k`, the softmax saturates, its Jacobian collapses toward zero, and gradients through the attention weights die. Dividing by `√d_k` restores unit variance.
- **Why the interviewer asks this:** It is the most-asked attention question at every level and the derivation is a clean test of whether you can do variance arithmetic.
- **Trap:** "It normalises the vectors." It does not — `q` and `k` are not renormalised, only the dot product is rescaled. Also note the scaling is by `√d_k`, not `d_k`; using `d_k` over-flattens the distribution and costs accuracy.

**Q10. What does multi-head attention buy you?**
- **Answer:** Instead of one attention distribution over `d_model` dimensions, you run `h` independent heads each of width `d_head = d_model/h` in parallel and concatenate. Each head produces its own convex mixture of values, so the layer's output is a *sum of h independent mixture distributions* rather than one — it can attend to several relationships simultaneously (syntactic head, coreference head, positional-neighbour head). Total parameter count is unchanged versus single-head of width `d_model`, and FLOPs are essentially unchanged.
- **Why the interviewer asks this:** Tests whether you know multi-head is not about capacity but about *diversity of routing*.
- **Trap:** "More heads = more parameters." No: `4 × d_model²` either way. More heads also means smaller `d_head`, so at some point `d_head` gets too small to represent a meaningful subspace — in practice `d_head = 64` or `128`.

**Q11. Why do transformers need positional encodings at all?**
- **Answer:** Self-attention is permutation-equivariant. `softmax(QKᵀ/√d_k)V` treats the input as a *set*; shuffle the tokens and the output is the shuffled same values. There is no built-in notion of order, so order has to be injected additively into the embeddings (or into Q/K, as RoPE does).
- **Why the interviewer asks this:** Fundamental architecture question, and the doorway to sinusoidal vs learned vs RoPE vs ALiBi.
- **Trap:** "Because the FFN is position-wise." The FFN being position-wise is a *consequence* of the same design, not the reason. The reason is attention's permutation symmetry.

**Q12. Describe sinusoidal positional encoding.**
- **Answer:** For position `pos` and dimension index `i`, `PE(pos, 2i) = sin(pos / 10000^{2i/d_model})` and `PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})`. It is added to the token embedding. Because `sin(a+b)` and `cos(a+b)` are linear functions of `(sin a, cos a)` for fixed `b`, the encoding of `pos + k` is a fixed linear map of the encoding of `pos` — so relative offsets are learnable by a linear projection.
- **Why the interviewer asks this:** Tests whether you know it is a *relative-offset* trick, not just a fancy fingerprint, and it is the setup for RoPE.
- **Trap:** Saying it is learned. Sinusoidal is fixed, parameter-free, and extrapolates (badly but non-trivially) beyond the training length.

**Q13. What is RoPE?**
- **Answer:** Rotary Position Embedding. Instead of adding a position vector to the embedding, RoPE rotates each consecutive pair of dimensions of `q` and `k` by an angle `θ_i = pos · 10000^{−2i/d}`. Since `R_m q · R_n k = qᵀ R_{n−m} k`, the resulting attention score depends only on the *relative* offset `m − n`, not on absolute position. It is applied inside attention, so it affects the score but not the value path.
- **Why the interviewer asks this:** RoPE is in Llama, Mistral, Qwen, Gemma — every modern open model. Not knowing it is a red flag for anyone claiming current LLM experience.
- **Trap:** "RoPE extends context for free." RoPE encodes relative position but was still *trained* at a fixed length; extending requires interpolation (NTK, YaRN, linear scaling) plus continued pretraining.

**Q14. Why is attention O(n²)?**
- **Answer:** The score matrix `QKᵀ` has `n × n` entries for a sequence of length `n`, and the softmax plus the `·V` multiply are both `O(n² d)`. Per layer the attention block costs about `4n² d_model` FLOPs. So doubling context quadruples attention cost while the FFN doubles.
- **Why the interviewer asks this:** The complexity claim is the standard lead-in to FlashAttention, KV cache, and SSM comparisons.
- **Trap:** "It's O(n²) in memory." It is O(n²) in FLOPs *and* O(n²) in the score matrix, but FlashAttention removes the O(n²) *memory* while leaving the FLOPs unchanged. Saying "FlashAttention makes attention O(n)" is a common rejection.

**Q15. What is the FFN in a transformer block, and how big is it?**
- **Answer:** Two position-wise linear layers with a nonlinearity: `FFN(x) = W_2 · act(W_1 x + b_1) + b_2`, with `d_ff ≈ 4 × d_model` (Llama-2 uses `d_ff = 11008` for `d_model = 4096`, i.e. 2.69×). It is applied identically to every position and holds roughly two thirds of a layer's parameters. It is where most of the model's factual knowledge is stored.
- **Why the interviewer asks this:** Parameter arithmetic is a standard screen, and it leads into the pre/post-LN and FFN-as-key-value-memory literature.
- **Trap:** Saying `4d` is universal. The 4× is the original Vaswani choice; Llama's SwiGLU uses `≈ 8/3 d_model` so that the three-matrix SwiGLU has the same parameter count as a two-matrix `4d` FFN.

**Q16. What is LayerNorm, and where do you put it in a block?**
- **Answer:** LayerNorm normalises a single token's activations across the feature dimension to zero mean and unit variance, then applies a learned scale and bias: `LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β`. It is not batch-dependent, so it works identically at batch size 1 and sequence lengths that vary. **Post-LN** (original transformer) puts it after the residual add; **Pre-LN** puts it inside the branch before the sublayer, leaving the residual stream unnormalised. Every modern LLM is Pre-LN or RMSNorm.
- **Why the interviewer asks this:** Pre vs Post LN is the clearest question that separates "I trained a transformer" from "I trained a transformer that converged".
- **Trap:** "LayerNorm and BatchNorm are the same but over different axes and both need batches." LayerNorm needs no batch statistics, so there is nothing to synchronise across GPUs and no train/eval discrepancy.

**Q17. What is a residual connection for, mathematically?**
- **Answer:** `y = x + F(x)` means `∂y/∂x = I + ∂F/∂x`. The identity term gives gradients a path to the input that is always 1, regardless of what `F` does — so an arbitrarily deep stack has a well-conditioned gradient route that never vanishes. Residuals are also why a transformer's residual stream acts as a shared communication bus across layers.
- **Why the interviewer asks this:** The `I +` is the answer. Candidates who say "it helps gradients flow" without the identity term have the intuition but not the mechanism.
- **Trap:** Calling it concatenation. It is element-wise addition — the instructor's own LSTM→transformer discussion blurs this, and interviewers notice.

**Q18. What is the hidden state of an RNN?**
- **Answer:** A single vector `h_t ∈ ℝ^{d_h}` that is the network's entire summary of the sequence up to and including step `t`. It is computed as `h_t = tanh(W_xh x_t + W_hh h_{t−1} + b_h)` and is overwritten at every step. Everything the model knows about the past must fit in those `d_h` numbers.
- **Why the interviewer asks this:** Sets up the information-bottleneck argument (L3 Q4), which is the deeper reason long-range dependencies break.
- **Trap:** Confusing the hidden state with the layer's output sequence. In a `return_sequences=True` stack you keep all `T` hidden states, but each one *individually* still only summarises its own prefix at fixed width.

**Q19. Why can't an RNN parallelise over time?**
- **Answer:** Because `h_t` is a strict function of `h_{t−1}`. The dependency is a serial data hazard, so no amount of hardware gets you `h_200` before you have `h_199`. In a transformer, every position's representation is computed from every other position simultaneously as a matrix product, so the whole sequence is one GEMM.
- **Why the interviewer asks this:** This is the *second* fatal problem, and many candidates only volunteer the gradient problem. Voluntarily raising it is a strong signal.
- **Trap:** "You could parallelise across the batch." True and irrelevant — the batch dimension was never the bottleneck; the time dimension is, and it is what caps GPU utilisation.

**Q20. What is truncated BPTT and why do you need it?**
- **Answer:** BPTT requires storing every `h_t` for the full sequence, so memory grows linearly in `T`; for long sequences it is infeasible and the gradient product is worthless past a few dozen steps anyway. Truncated BPTT splits the sequence into segments of length `k`, carries the final state forward as a *constant* (detached, no gradient), and backprops only within each segment. Memory becomes `O(k)`.
- **Why the interviewer asks this:** It is the practical admission that the architecture cannot learn the long-range structure you wanted, and it is a fixture of every RNN codebase.
- **Trap:** "TBPTT learns long-range dependencies over many segments." It does not. Gradients never cross a segment boundary; only *state* does. Long-range credit assignment is limited to `k` steps, by construction.

**Q21. What is gradient clipping and when is it mandatory?**
- **Answer:** Rescaling the gradient when its global norm exceeds a threshold: `g ← g · threshold / max(threshold, ‖g‖)`. In PyTorch, `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. It is mandatory for RNNs/LSTMs because the same product that causes vanishing causes exploding, and exploding gradients produce NaN weights from which training never recovers. Llama-family LLM training also clips at 1.0, so it did not disappear with attention.
- **Why the interviewer asks this:** Both a practical hygiene check and a trap: many candidates say clipping "solves" vanishing gradients.
- **Trap:** "Clipping fixes the vanishing side." It only caps the upper bound. It does nothing for a gradient that is `10⁻¹⁵` while a sibling term is `10⁻²`.

**Q22. Why was there no pretrained checkpoint to fine-tune in the RNN era?**
- **Answer:** Three reasons, all structural. (1) With no attention, a model's learned features are entangled with the tokenizer's exact vocabulary and the task's exact label space, so there was no transferable substrate. (2) Training was slow enough that a large general-purpose corpus run was not affordable at research scale. (3) The gradient problem meant a generic pretraining objective (language modelling) did not produce representations that survived to a downstream task without task-specific re-tuning. ELMo (2018) was the first real counterexample — and it was LSTM-based.
- **Why the interviewer asks this:** It is the bridge from the architecture story to the fine-tuning story, which is what the whole handbook is about.
- **Trap:** "There were no pretrained models until BERT." ELMo (Feb 2018), ULMFiT (May 2018) and GPT-1 (Jun 2018) are all pre-BERT, and the first two are LSTM-based.

**Q23. What is ULMFiT and why does it matter to this topic?**
- **Answer:** Universal Language Model Fine-tuning (Howard & Ruder, 2018) — a 3-layer AWD-LSTM with a task-specific classifier head, trained with a language-model objective on a general corpus and then fine-tuned per task using discriminative learning rates, slanted triangular schedules, and gradual unfreezing. It is **not** an encoder-decoder and uses **no attention**. It cut error by 18–24 % on six text-classification datasets and matched a model trained on 100× more data using 100 labelled examples.
- **Why the interviewer asks this:** It is the standard "gotcha" — it disproves the folk claim that transfer learning required transformers.
- **Trap:** Calling it a transformer or an attention model. The instructor's own video is loose here; the architecture is a 3-layer LSTM.

**Q24. What is a KV cache and what does it cost?**
- **Answer:** At inference the keys and values for already-generated tokens never change, so you store them instead of recomputing the prefix at every step. Per token the cache is `2 × n_layers × n_kv_heads × d_head × bytes`. For Llama-3-8B in fp16 that is `2 × 32 × 8 × 128 × 2 = 131 072` bytes = **128 KB per token**, i.e. 16 GB for a 128 k context. It is the reason MQA/GQA exist.
- **Why the interviewer asks this:** The KV cache is where the quadratic cost moves from FLOPs to memory, and it is the most common production serving bottleneck.
- **Trap:** Forgetting the factor of 2 (keys *and* values). Halving the answer is a visible arithmetic error.

**Q25. What is FlashAttention, in one line?**
- **Answer:** An IO-aware exact attention kernel that tiles `Q`, `K` and `V` through on-chip SRAM and uses an online-softmax running rescale, so the `n × n` score matrix is never materialised in HBM. Memory drops from `O(n²)` to `O(n)` and wall-clock improves 2–4×, **with identical FLOPs and identical outputs**.
- **Why the interviewer asks this:** It tests whether you understand the memory hierarchy as the real constraint, not FLOPs.
- **Trap:** "FlashAttention makes attention linear." It is exact attention; the arithmetic is unchanged at `O(n²)`. It makes attention *memory*-linear and *faster*, not asymptotically cheaper.

**Q26. What is Mamba / an SSM?**
- **Answer:** A state-space model: a linear time-invariant recurrence `h_t = Ā h_{t−1} + B̄ x_t`, `y_t = C h_t`, discretised from a continuous system. Because the recurrence is linear and time-invariant, it can be computed as a global convolution for training (parallel over `n`) and as a recurrence for inference (`O(1)` state per step, no KV cache). Mamba/S6 makes `B`, `C` and the step size `Δ` input-dependent ("selective"), which is what lets it filter content rather than just smooth it.
- **Why the interviewer asks this:** The modern challenger to attention. Every 2025-era LLM-infra interview has some version of it.
- **Trap:** "Mamba replaced the transformer." It replaced the *attention block*, but Mamba blocks still use gated MLP/FFN blocks and residual streams; and pure Mamba underperforms transformers on recall-heavy tasks, which is why every production hybrid (Jamba, Zamba, Samba, Hymba, Qwen3-Next) interleaves attention layers with SSM layers.

**Q27. Give the parameter-count formula for a transformer layer.**
- **Answer:** Attention: `4 d_model²` (`W_Q, W_K, W_V, W_O`, each `d_model × d_model`, ignoring biases). FFN with expansion `r`: `2 r d_model²` — or `3 r d_model²` for a three-matrix SwiGLU. LayerNorms: `4 d_model`. So a layer is roughly `4d² + 2rd²` = `12 d_model²` at `r = 4`.
- **Why the interviewer asks this:** Back-of-envelope sizing is a standard screen. It comes up in every LoRA rank / VRAM estimate.
- **Trap:** Forgetting the embedding table. It is `vocab × d_model` and is often 5–10 % of the total (Llama-2-7B: 32000 × 4096 = 131 M, 1.94 % of the model because tied embeddings are counted once).

**Q28. What is the difference between a tokenizer and a vocabulary?**
- **Answer:** The tokenizer is the algorithm plus its learned merge table (BPE, WordPiece, Unigram); the vocabulary is the resulting fixed index→string mapping. Changing the tokenizer changes the token IDs, which changes the meaning of every row of the embedding matrix. This is precisely why you cannot swap tokenizers on a pretrained model.
- **Why the interviewer asks this:** It is the mechanism behind the notebook's silent failure (L2 Q20) and behind a very common production incident when someone re-trains a tokenizer on domain data.
- **Trap:** "The tokenizer is part of the model, so `save()` keeps it." Keras `model.save()` does not save a tokenizer; HuggingFace `save_pretrained()` does. That asymmetry has burned countless projects.

**Q29. What is teacher forcing?**
- **Answer:** In sequence-to-sequence training you feed the *ground-truth* previous token as the decoder input at each step, instead of the model's own previous prediction. It makes training parallel and stable, at the cost of **exposure bias**: at inference the model sees its own (possibly wrong) outputs, a distribution it never trained on. Scheduled sampling anneals between the two.
- **Why the interviewer asks this:** It is the seq2seq-era default that the notebook uses, and it is the origin of the "the model works in the lab, not in production" class of failures.
- **Trap:** "Teacher forcing is required for RNN decoders." It is a choice. The alternative — feeding back predictions — is slower per step but eliminates exposure bias.

**Q30. What is the difference between hidden-state size and context length?**
- **Answer:** Hidden size `d_h` is the *width* of the state vector; context length is how many tokens the model can condition on. In an RNN they are coupled through a bottleneck: the entire past is compressed into `d_h` numbers, so a 256-dimensional state carrying a 1000-token history is 256 floats for 4000+ tokens of information. In a transformer the context is a sequence of `n` vectors and no compression is forced — the model can address any of them directly, which is why scaling context length is a matter of compute and memory rather than of representational capacity.
- **Why the interviewer asks this:** It surfaces the bottleneck argument, which is the deepest reason RNNs lose, and it shows whether you think of attention as *uncompressed memory*.
- **Trap:** "Just make the hidden state bigger." The state is `d_h`, but the parameters grow as `d_h²` and the optimisation problem gets harder, so you cannot buy your way out linearly.

---

## Level 2 — Applied & Implementation

**Q1. Write the forward equations of a vanilla RNN.**
- **Answer:**
  `a_t = W_xh x_t + W_hh h_{t−1} + b_h`; `h_t = tanh(a_t)`; `ŷ_t = softmax(W_hy h_t + b_y)`.
  Shapes for the CS-05 reference configuration: `x_t ∈ ℝ^{128}`, `h_t ∈ ℝ^{256}`, `W_xh ∈ ℝ^{256×128}`, `W_hh ∈ ℝ^{256×256}`, `W_hy ∈ ℝ^{V×256}`. Parameters: `256·128 + 256·256 + 256 = 32 768 + 65 536 + 256 = 98 560`, plus the output layer.
- **Why the interviewer asks this:** Notation check — if your `W_hh` is `256×256` and your `h` is 256, you and the interviewer can actually talk.
- **Trap:** Writing `h_t = tanh(W [x_t; h_{t−1}])` without realising that is the *same thing* as the two-matrix form only when the concatenated matrix is block-partitioned; interviewers use the two-matrix form because it exposes which term carries the time dependency.

**Q2. Write the LSTM gate equations.**
- **Answer:**
  `f_t = σ(W_f [h_{t−1}; x_t] + b_f)`
  `i_t = σ(W_i [h_{t−1}; x_t] + b_i)`
  `g_t = tanh(W_g [h_{t−1}; x_t] + b_g)`
  `o_t = σ(W_o [h_{t−1}; x_t] + b_o)`
  `C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t`
  `h_t = o_t ⊙ tanh(C_t)`
  Each `W_· ∈ ℝ^{d_h × (d_h + d_x)}`, so `4 d_h (d_h + d_x)` parameters plus `4 d_h` biases.
- **Why the interviewer asks this:** Canonical whiteboard task; also the setup for the `diag(f_t)` derivation.
- **Trap:** Putting the nonlinearity on `C_t` inside the recurrence. `C_t` accumulates *linearly*; the only nonlinearity on the highway is the multiplication by the sigmoid gates. Add a `tanh` around the whole update and you destroy the highway and the gradient argument with it.

**Q3. Write the BPTT gradient of the loss with respect to `W_hh`.**
- **Answer:**
  `∂L/∂W_hh = Σ_{t=1}^{T} Σ_{k=1}^{t} [ ∂L_t/∂h_t · Π_{j=k+1}^{t} diag(1 − h_j²) W_hhᵀ ] h_{k−1}ᵀ`
  The outer sum is over output steps, the inner sum over the source steps that influence them, and the bracketed product is the Jacobian chain. Note the `h_{k−1}ᵀ` outer product on the right — that is the standard "gradient × input" form for a weight matrix.
- **Why the interviewer asks this:** This is *the* question for this module. A candidate who can write it can answer everything else by reading terms off it.
- **Trap:** Writing a single product instead of a double sum. The gradient at step `t` receives contributions from *every* `k ≤ t`, and it is the relative sizes of those terms that matter — not the size of any one of them.

**Q4. Why does Xavier initialisation put an RNN at the edge of stability?**
- **Answer:** Xavier sets `Var(W) = 1/fan_in`, so for `W_hh ∈ ℝ^{d_h × d_h}` the entries are drawn with standard deviation `1/√d_h`. The largest singular value of a `d_h × d_h` random matrix with i.i.d. entries of variance `σ²` concentrates near `σ(√d_h + √d_h) = 2σ√d_h` for the full Marchenko–Pastur edge, but the *typical* spectral radius of a square Gaussian matrix is `σ√d_h`. With `σ = 1/√d_h` that is exactly **1.0**. So the default init places the RNN's Jacobian norm — and therefore the gradient product — right at the critical value where it is as likely to explode as to vanish, and any drift in `‖W_hh‖` during training pushes it off the knife edge.
- **Why the interviewer asks this:** It connects initialisation theory to a practical RNN training failure, and it is a favourite of research-leaning loops. `[Company style: big-tech research]`
- **Trap:** "Xavier init solves the vanishing gradient problem." It sets the *initial* spectral radius near 1; it provides no mechanism to keep it there over thousands of steps.

**Q5. Compute the gradient decay for a dependency 50 steps away, at `λ = 0.9`.**
- **Answer:** `λ^50 = 0.9^50 = e^{50 ln 0.9} = e^{50 × (−0.10536)} = e^{−5.268} = 5.16 × 10⁻³`. At `λ = 0.8`: `e^{50 × (−0.22314)} = e^{−11.157} = 1.42 × 10⁻⁵`. At `λ = 0.5`: `e^{−34.66} = 8.9 × 10⁻¹⁶` — below fp32 machine epsilon (`1.19 × 10⁻⁷`), so the term is not merely small, it is *unrepresentable*. And since `∂L/∂h_0` is dominated by the `k = t` term (which is `O(1)`), the long-range contribution is not just lost in absolute terms, it is lost relative to the other terms summed into the same gradient.
- **Why the interviewer asks this:** Anyone can say "the gradient decays geometrically". Being asked to produce the number live filters for people who have actually looked at a gradient-norm plot.
- **Trap:** Answering `0.9^50 ≈ 0.9` because "50 steps is not that many". Do the exponential.

**Q6. Adam is scale-invariant. Why doesn't that fix vanishing gradients?**
- **Answer:** Adam normalises by the running RMS of each parameter's gradient, `m̂/(√v̂ + ε)`. Because the update is invariant to multiplying the gradient by a constant, a *uniformly* small gradient is recovered: `c·g/√(c²v) = g/√v`. But vanishing gradients are not uniform — inside a single parameter's gradient you have `Σ_t Σ_k` where one term is `O(1)` and another is `10⁻¹⁵`. Adam rescales the *sum*, which is dominated by the large term; the long-range term is numerically gone and no per-parameter rescale can recover it. Worse, `ε = 1e-8` in the denominator sets a floor: once `√v̂ < ε`, the update degenerates back to raw `m̂` and the scale invariance is lost.
- **Why the interviewer asks this:** It is the most common *wrong* answer in the whole topic. Senior candidates are expected to know this cold.
- **Trap:** "Adam fixes it because it adapts per-parameter learning rates." Adaptivity is per-*parameter*, not per-*pathway*. It cannot distinguish two terms inside one scalar gradient.

**Q7. Why is `diag(f_t)` a better gradient path than `W_hhᵀ`?**
- **Answer:** Three structural properties. (1) It is **diagonal**, so there is no mixing between state dimensions — a gradient flowing back through dimension `i` is multiplied only by `f_{t,i}` and cannot be amplified by off-diagonal weight mass. (2) Its entries lie in `(0,1)` by construction (sigmoid output), so it can shrink but cannot explode. (3) There is no weight matrix in the path at all, so the path's norm is not tied to `‖W‖` being near 1 — the RNN's knife-edge init problem simply does not apply.
- **Why the interviewer asks this:** It is the real content of "gradient highway", stated precisely.
- **Trap:** "So the gradient never vanishes in an LSTM." It can, and does — see the next question.

**Q8. What are the four reasons the LSTM still does not solve long-range dependency?**
- **Answer:**
  1. **`f_t` is learned and must be able to drop below 1.** To forget anything, the network must set `f_t < 1`; over 500 steps a forget gate of 0.99 gives `0.99^{500} = 0.0066`. Information that must persist cannot be protected from the gate that must also be able to delete.
  2. **The `h`-paths are still RNN paths.** `h_t = o_t ⊙ tanh(C_t)` and the gates all read `h_{t−1}`, so every gradient that flows through the gates (rather than through `C`) still carries `W_·ᵀ` matrices and `σ'`/`tanh'` factors, and `σ'(·) ≤ 0.25`.
  3. **Read-out saturation.** `h_t = o ⊙ tanh(C_t)` saturates for large `|C_t|`, so even a perfectly preserved cell value is attenuated (with vanishing derivative) when it is read out.
  4. **Clipping is still mandatory** because the exploding side persists, and clipping introduces its own bias — it rescales the whole gradient, including the well-behaved short-range terms.
- **Why the interviewer asks this:** This is the question that separates candidates who have read the LSTM paper's abstract from those who have thought about why LSTMs still needed truncated BPTT. `[Company style: big-tech research]`
- **Trap:** "LSTMs solve vanishing gradients." They *mitigate* them. The gap between "mitigate" and "solve" is exactly why attention won.

**Q9. Derive the `1/√d_k` scaling.**
- **Answer:** Let `q, k ∈ ℝ^{d_k}` have i.i.d. components with `E[q_i] = E[k_i] = 0` and `Var(q_i) = Var(k_i) = 1`. Each product term `q_i k_i` then has mean `0` and variance `1·1 = 1`. The dot product is a sum of `d_k` independent such terms, so `Var(q·k) = Σ Var(q_i k_i) = d_k`, giving `sd(q·k) = √d_k`. At `d_k = 128` that is `≈ 11.3`, so the softmax logits scatter over roughly ±11 — saturating, with near-one-hot outputs and a Jacobian near zero. Scaling by `1/√d_k` restores unit standard deviation.
- **Why the interviewer asks this:** Complete derivation, three lines, no hand-waving available. Asked at essentially every level for attention roles.
- **Trap:** Saying "it prevents the softmax from being too peaked" without the variance argument. Also: if your `q` and `k` are not unit-variance (they come from learned projections after a LayerNorm, so they roughly are), the derivation's assumptions are worth stating out loud.

**Q10. Compute the attention parameter count for `d_model = 4096`.**
- **Answer:** `W_Q, W_K, W_V, W_O` are each `4096 × 4096 = 16 777 216` params, so attention is `4 × 16 777 216 = 67 108 864`. For Llama-2-7B, `W_K` and `W_V` are not `4096×4096`: with 32 heads of `d_head = 128` and 32 KV heads they are full width, so the count stands. (`n_kv_heads × d_head × d_model` per projection.)
- **Why the interviewer asks this:** The follow-up "and the FFN?" is where you show you can size a model on a whiteboard.
- **Trap:** Forgetting `W_O`. Dropping the output projection is the most common error and gives 25 % of the correct answer.

**Q11. Pre-LN vs Post-LN — what actually changes?**
- **Answer:** Post-LN computes `LN(x + F(x))`; Pre-LN computes `x + F(LN(x))`. Post-LN leaves no unnormalised identity path, so the gradient through the residual stream is scaled by the LayerNorm Jacobian at every block and the effective update magnitude grows with depth. That is why Post-LN transformers need learning-rate warmup over thousands of steps — the original paper used 4000 — and diverge without it. Pre-LN keeps `I + ∂F/∂x` clean, trains without warmup, and tolerates much higher learning rates; the price is a growing activation variance with depth, which is why modern models add a final LayerNorm (or use RMSNorm with a scaled init).
- **Why the interviewer asks this:** It is the clearest "have you actually trained one" question in the architecture section.
- **Trap:** "Pre-LN is strictly better." Pre-LN models are sometimes slightly worse at a *matched* step count early in training; they win on stability and on the ability to use large learning rates without a long warmup.

**Q12. Do the Llama-2-7B parameter arithmetic.**
- **Answer:** `vocab × d_model = 32000 × 4096 = 131 072 000` (tied embeddings). Each of 32 layers: attention `4 × 4096² = 67 108 864`; FFN SwiGLU with `d_ff = 11008` and three matrices `3 × 4096 × 11008 = 135 266 304`; two RMSNorm vectors `2 × 4096 = 8192`. Total per layer `202 383 360`. Times 32 = `6 476 267 520`. Plus embeddings `131 072 000`, plus the final norm `4096`. Total `6 607 343 616` — the published count is `6 738 415 616`, and the difference is the untied output projection (`32000 × 4096 = 131 072 000`): `6 607 343 616 + 131 072 000 = 6 738 415 616`. **Exact match.** The split is FFN 64.2 %, attention 31.9 %, embeddings 3.89 % (tied).
- **Why the interviewer asks this:** Being able to land on a published parameter count to the digit is a strong competence signal, and it lets them ask "so where does LoRA go?" next. `[Company style: big-tech research]`
- **Trap:** Forgetting the untied output projection and concluding the model is 6.6 B. Llama-2 unties it; Llama-3 does not (it ties), which is the kind of detail that distinguishes the two families.

**Q13. Compute the KV cache for a given model and context.**
- **Answer:** `KV bytes = 2 × n_layers × n_kv_heads × d_head × bytes_per_element × tokens`. Llama-3-8B (`n_layers = 32`, `n_kv_heads = 8` via GQA, `d_head = 128`) in fp16: `2 × 32 × 8 × 128 × 2 = 131 072` bytes = **128 KB/token** → 16 GB at 128 k tokens, 2 GB at 16 k. Llama-2-7B (`n_kv_heads = 32`, no GQA): `2 × 32 × 32 × 128 × 2 = 524 288` = **512 KB/token** → 64 GB at 128 k, which is why GQA exists.
- **Why the interviewer asks this:** Serving interviews live here. It also tests whether you know MQA/GQA is a *KV cache* optimisation, not a FLOP optimisation.
- **Trap:** Computing only keys. The factor of 2 is not optional, and forgetting it produces a plausible-but-half answer that experienced interviewers catch immediately.

**Q14. At what sequence length does attention cost more than the FFN?**
- **Answer:** Attention is `≈ 4n²d_model` FLOPs per layer; the FFN is `≈ 2 · n · d_model · d_ff = 8 n d_model²` at `d_ff = 4d_model`. Setting them equal: `4n²d = 8nd²` → `n = 2d_model`. With SwiGLU's `d_ff = 8/3 d_model` and three matrices, the FFN is `≈ 6 n d_model²` and the crossover is `n = 1.5 d_model`. So for `d_model = 4096` the crossover is around `n ≈ 6000–8000`; below that the FFN dominates the FLOP budget, above it attention does. This is why long-context work is attention-kernel work.
- **Why the interviewer asks this:** It shows you can decompose a transformer's cost model, which is the basis of every context-length and serving decision.
- **Trap:** Assuming attention dominates at all lengths. At `n = 512` with `d_model = 4096` the FFN is more than 8× the attention cost.

**Q15. What is arithmetic intensity, and why does an RNN run at 1 FLOP/byte?**
- **Answer:** Arithmetic intensity is FLOPs performed per byte moved from memory. A **matrix-vector** product reads `d²` weights and does `2d²` FLOPs → `≈ 1 FLOP/byte` (2 FLOPs per 2-byte fp16 weight). A **matrix-matrix** product with batch `n` reads the same `d²` weights but does `2nd²` FLOPs → `n` FLOPs/byte. An A100 with ~2 TB/s of HBM bandwidth and ~312 TFLOP/s of fp16 tensor throughput needs `312e12/2e12 ≈ 156` FLOP/byte to saturate. So an RNN step (`n = 1`) runs at about 1/156 of peak — under 1 % — while a transformer GEMM with `n ≥ 156` saturates the machine.
- **Why the interviewer asks this:** This is the *quantitative* form of "RNNs can't parallelise", and it is the roofline model in one sentence. Strong signal for inference/serving roles. `[Company style: big-tech research]`
- **Trap:** Framing the problem as FLOPs. An RNN and a transformer can do *identical* FLOPs and differ 100× in wall-clock, because the constraint is bandwidth and kernel-launch overhead, not arithmetic.

**Q16. Why does an LSTM have four times the parameters of a vanilla RNN?**
- **Answer:** The LSTM has four weight matrices of the same shape as the RNN's `W_xh`/`W_hh` pair — one each for `f`, `i`, `g`, `o`. For `d_x = 128`, `d_h = 256`: each gate is `256 × (256 + 128) = 98 304` params plus `256` bias, so `4 × (98 304 + 256) = 394 240`. A vanilla RNN with the same widths is `256 × 128 + 256 × 256 + 256 = 98 560`. Ratio `394 240 / 98 560 = 4.0`.
- **Why the interviewer asks this:** It is the fastest way to check whether someone has actually looked at a `model.summary()`. The notebook's LSTM line is exactly `394,240`.
- **Trap:** `4×` only holds when `d_x = d_h`. In general the ratio is `4(d_h + d_x)/(d_h + d_x) = 4` for the recurrent part but the bias counts differ; the clean 4.0 above is the reference configuration.

**Q17. GRU vs LSTM at `d_h = 256`, `d_x = 128` — do the arithmetic.**
- **Answer:** GRU has three matrices (`W_z`, `W_r`, `W_h`) of shape `d_h × (d_h + d_x)` plus three bias vectors: `3 × (256 × 384) + 3 × 256 = 3 × 98 304 + 768 = 295 680 + 768 = 296 448`. LSTM: `4 × 98 304 + 4 × 256 = 393 216 + 1024 = 394 240`. GRU is **24.8 % smaller** (`296 448 / 394 240 = 0.752`).
- **Why the interviewer asks this:** Cheap arithmetic that tests genuine familiarity, and the ratio is a nice round quarter.
- **Trap:** "GRU has 2/3 of the LSTM's parameters because it has 2 of 3 gates." Wrong fraction — two states' worth of machinery collapse into one, so it is 3 weight matrices vs 4, i.e. 3/4.

**Q18. How do you choose the truncation length for TBPTT?**
- **Answer:** Match it to the longest dependency the task actually requires, then verify empirically. If your signal is "the label depends on a token within the last 40 steps", `k = 64` suffices and anything longer just costs memory and time. If you genuinely need a 200-step dependency you cannot get it from TBPTT at `k = 200` on real hardware — the honest answer is that the architecture, not the hyperparameter, is the limit. Practical defaults: `k = 32–128` for tagging, `k = 100–200` for sequence classification on short texts. Carry `h` forward with `detach()` between segments.
- **Why the interviewer asks this:** It tests whether you understand that TBPTT is a *cap on credit assignment*, not a memory trick with no cost.
- **Trap:** "Set `k` to the full sequence length." At `T = 1000` and `d_h = 512` with batch 64 that is `1000 × 64 × 512 × 4 B = 131 MB` per layer just for activations, before the gates — and the gradient product at that depth is worthless anyway.

**Q19. `padding='pre'` vs `'post'` for an LSTM — does it matter?**
- **Answer:** Enormously, for any model that reads only the final hidden state. With `padding='post'` (the notebook's choice) the padding tokens come *after* the real content, so the last `h_T` is a function of `<PAD>` tokens, and the classifier head is reading a state that has been overwritten by noise. With `padding='pre'` the real content ends at the final step and `h_T` summarises the review. The alternative fixes are `Masking()` before the LSTM (which makes the layer skip padded steps) or `Bidirectional` pooling over all timesteps. A ~3–5 accuracy-point swing from this one flag is a real, measurable effect.
- **Why the interviewer asks this:** It is the most common silent bug in RNN text classification, and it is directly visible in the companion notebook.
- **Trap:** "Keras pads with zeros so the LSTM ignores them." A zero input still produces `h_t = tanh(W_hh h_{t−1} + b)` — nonzero, and it moves the state. Only an explicit `Masking` layer (or `mask_zero=True` on the embedding) skips them.

**Q20. The notebook reports `loss: 8.9872` on a fresh seq2seq task. What does that number mean?**
- **Answer:** The decoder vocabulary is 8000 and the targets are `np.random.randint(1, 8000, ...)` — uniform noise. A model that has learned nothing but the marginal distribution assigns probability `1/8000` to every token, giving cross-entropy `−ln(1/8000) = ln 8000 = 8.9871968...`. The reported `8.9872` **is** `ln 8000` to four decimal places, and the accuracy `1.2756 × 10⁻⁴ ≈ 1/8000 = 1.25 × 10⁻⁴`. The model did not learn to translate; it learned to output the uniform distribution, which is the optimal solution to an unlearnable target. The run is a perfect negative control.
- **Why the interviewer asks this:** It is the single best "do you read your own loss curves" question in the module. Recognising `ln(V)` on sight is a core diagnostic reflex.
- **Trap:** "Loss 8.98 means the model is broken." The model is fine; the *data* is random. Diagnosing "loss stuck at ln(V)" as a data/label problem rather than a model problem is the correct call.

**Q21. What is the difference between a plain encoder-decoder and attention-based seq2seq?**
- **Answer:** A plain encoder-decoder passes a single fixed-size vector — the encoder's final state — as the decoder's initial state. Every bit of the source must be compressed into `d_h` numbers, and quality collapses for sources much longer than the decoder's first few steps. Attention-based seq2seq (Bahdanau 2014, Luong 2015) instead gives the decoder a *different* context vector at every decoding step: a convex mixture of all encoder states, re-weighted per step. The source no longer has to be compressed, and the path from any source token to the loss is short. Attention was invented as an RNN accessory, not as a replacement.
- **Why the interviewer asks this:** It shows attention's actual origin and pre-empts the misconception that attention arrived with the transformer.
- **Trap:** "Attention came from `Attention Is All You Need`." It came from neural machine translation two years earlier, bolted onto LSTMs.

**Q22. You see training loss pinned at exactly `ln(vocab_size)` from step 0. List your checks in order.**
- **Answer:**
  1. **Are the targets random / misaligned?** Compute `p(y)` empirically on the label tensor. If it is uniform, you have the notebook's bug. (Fastest, most common.)
  2. **Is the loss reduction right?** `CrossEntropyLoss` in PyTorch expects raw logits; applying `softmax` first gives it probabilities and produces a near-constant high loss. `reduction='mean'` over a padded batch without `ignore_index=-100` dilutes with padding.
  3. **Is the label shift correct?** For next-token prediction, `input = tokens[:-1]`, `target = tokens[1:]`. Off-by-one produces a loss that stays near `ln(V)` forever.
  4. **Are the logits all equal?** Print `logits.std()`; if it is `~1e-6`, the head is dead (bad init, or a frozen/zeroed weight).
  5. **Is the LR zero or the optimizer not stepping?** Check `param.grad` is not `None` and that the optimizer was constructed *after* the parameters.
- **Why the interviewer asks this:** Ordered debugging is the L5 skill; this question is the bridge from L2 to L5. The order matters more than the list.
- **Trap:** Starting with the learning rate. `ln(V)` from step 0 is a data/wiring signature, not an optimisation signature — no learning rate produces exactly `ln(V)` immediately.

**Q23. RoPE vs ALiBi — when do you use which?**
- **Answer:** RoPE encodes relative position as a rotation of `q` and `k`; it is the default in Llama/Mistral/Qwen/Gemma, has no learned parameters beyond the base frequency, and extends with interpolation (linear, NTK, YaRN) plus continued pretraining. ALiBi adds a fixed linear bias `−m · (i − j)` to the attention scores — zero parameters, trained-length extrapolation that degrades gracefully, but it has largely lost adoption because it cannot be combined with the RoPE-based tooling and it does not match RoPE's quality at matched scale. Practical rule in 2025: **use RoPE and a context-extension recipe; use ALiBi if you specifically need length extrapolation with no fine-tuning and can accept the quality ceiling.**
- **Why the interviewer asks this:** Both are in production codebases; the answer reveals whether you have opinions grounded in the literature or just recall of acronyms.
- **Trap:** "ALiBi extrapolates perfectly." It extrapolates *gracefully*, with measurable degradation, and only for lengths a modest multiple of training length.

**Q24. Why is the FFN roughly two thirds of the parameters?**
- **Answer:** Attention is `4 d_model²`; the FFN at `d_ff = 4d_model` is `2 × 4 d_model² = 8 d_model²`, which is `8/12 = 66.7 %` of the layer. Llama's SwiGLU with `d_ff = 8/3 d_model` and three matrices is `3 × 8/3 d_model² = 8 d_model²` — the same, which is why the 8/3 constant was chosen. Empirically the FFN is where factual associations live (the "FFN as key-value memory" reading), which is why knowledge-editing methods like ROME target `W_1`/`W_2` rather than attention, and why removing FFN capacity damages knowledge tasks more than reasoning tasks.
- **Why the interviewer asks this:** Parameter budgeting shows up in every LoRA-rank and VRAM conversation.
- **Trap:** "The FFN is just a nonlinearity." It is the model's largest parameter store and its main knowledge substrate.

**Q25. What are MQA and GQA, and why do they matter?**
- **Answer:** Both shrink the KV cache by sharing key/value heads. **MQA** uses one KV head for all query heads — a `n_heads`-fold cache reduction, with quality loss and a tendency to unstable training. **GQA** groups query heads into `g` groups, each with its own KV head — e.g. Llama-3-8B has 32 query heads and 8 KV heads, a 4× reduction in cache (512 → 128 KB/token) at near-MHA quality. It does **not** reduce attention FLOPs meaningfully; it reduces memory and memory bandwidth, which at long context is the actual bottleneck.
- **Why the interviewer asks this:** Every modern open model uses GQA; not knowing it means you have not read a config file.
- **Trap:** "GQA makes attention cheaper." It makes the *cache* cheaper. The `QKᵀ` FLOPs are unchanged.

**Q26. How do you decide between an SSM/hybrid and a transformer?**
- **Answer:** Three axes. (1) **Task type:** if it requires precise recall of arbitrary tokens from far back — long-document QA with exact spans, in-context retrieval, code with distant references — pure SSMs are weaker; hybrid interleaving (attention every 4th–8th layer) recovers most of it. (2) **Sequence length:** SSM cost is linear in `n` and constant-state at inference, so the longer the sequence the more the SSM wins on memory; below ~8 k the transformer's quality advantage usually dominates. (3) **Serving profile:** SSMs have no KV cache, so their per-token decode memory is constant and their throughput at high concurrency is far better. Default in 2025: transformer unless you are at ≥100 k context or high-concurrency streaming, in which case a hybrid.
- **Why the interviewer asks this:** It is the "what's next" question and it rewards reading beyond the course. `[Company style: startup ML eng]`
- **Trap:** Treating it as an either/or. Every production "Mamba model" is a hybrid — Jamba is roughly 1 attention layer per 7 Mamba layers.

**Q27. What does `LSTM(latent_dim, return_state=True)` actually return?**
- **Answer:** Three tensors: the full output sequence `(batch, T, latent_dim)`, the final hidden state `state_h` `(batch, latent_dim)`, and the final cell state `state_c` `(batch, latent_dim)`. `return_sequences` controls only the first. In Keras 3 / TF 2.x the return order for `LSTM` is `[output, state_h, state_c]`; for `GRU` it is `[output, state_h]` (no cell). `state_c` is what you must carry across TBPTT segments *and* across decoder steps in a seq2seq model — dropping it means the decoder starts every step with an empty memory.
- **Why the interviewer asks this:** It is exactly the API detail the notebook depends on, and confusion between `h` and `C` is a classic.
- **Trap:** Unpacking as `output, state_h = model(...)` and silently assigning `state_c` to nothing. The notebook does `lstm_layer, state_h, state_c = ...` and then uses `state_h` only for the classifier — correct there, but only because the classifier needs no `C`.

**Q28. What are the silent failure modes of an underfit fine-tune?**
- **Answer:** The dangerous ones produce plausible outputs:
  - **Loss looks fine, task accuracy is at chance.** Binary cross-entropy near `ln 2 = 0.693` is exactly the notebook's `0.6942` — the model predicts the base rate and the metric hides it.
  - **The model learned the format, not the task.** It emits well-formed JSON with wrong values; token-level accuracy is high, exact-match is zero.
  - **Vocabulary mismatch.** Reloading the tokenizer at a smaller `num_words` maps rare tokens to `<UNK>`, so the frozen embedding receives a distribution it never trained on and the model degrades *silently*, with a small loss increase rather than an error.
  - **Truncation.** `maxlen=200` with `truncating='post'` discards the review's conclusion, so the label is genuinely unpredictable from the input — the model is right to be at chance.
  - **Frozen layers that should be trainable (or vice versa).** `trainable=False` on the wrong prefix means nothing updates; `model.summary()` does not show which layers are frozen, so you must count `requires_grad` parameters explicitly.
- **Why the interviewer asks this:** The brief's whole point — a fine-tune that "runs" tells you nothing. This question is the practical payoff of the module.
- **Trap:** Reporting only accuracy. Always report the loss against the base-rate baseline (`ln 2` for balanced binary, `ln V` for language modelling). A metric without a baseline is not a result.

---

## Level 3 — Advanced, Internals & Theory

**Q1. Prove the geometric bound on the RNN Jacobian product.**
- **Answer:** Start from `h_j = tanh(W_hh h_{j−1} + W_xh x_j + b)`. By the chain rule,
  `∂h_j/∂h_{j−1} = diag(tanh'(a_j)) · W_hhᵀ`, where `tanh'(a) = 1 − tanh²(a) = 1 − h_j²`.
  Take any submultiplicative norm:
  `‖∂h_j/∂h_{j−1}‖ ≤ ‖diag(1 − h_j²)‖ · ‖W_hhᵀ‖ ≤ γ · ρ`
  with `γ = max_j ‖diag(1 − h_j²)‖∞ = max_j (1 − h_j²) ≤ 1` (equality only at `h_j = 0`) and `ρ = σ_max(W_hh)`. Applying submultiplicativity across the product:
  `‖∂h_t/∂h_k‖ = ‖Π_{j=k+1}^{t} ∂h_j/∂h_{j−1}‖ ≤ (γρ)^{t−k}`.
  If `γρ < 1` the bound decays geometrically; if `γρ > 1` it grows geometrically — exploding. The critical case `γρ = 1` is a random walk, not stability. Note that `γ ≤ 1` is what makes vanishing the *default* outcome: the nonlinearity alone contributes at most 1 per step, so `ρ` must exceed 1 just to break even, and gradient descent has no term pushing it there.
- **Why the interviewer asks this:** Full derivation with explicit constants. This is the theorem the whole topic rests on.
- **Trap:** Using `‖W_hhᵀ‖₂ = σ_max` and then claiming `ρ < 1` guarantees vanishing. Vanishing is guaranteed only when `γρ < 1`; and because `γ` depends on the data through `h_j`, the *effective* contraction rate is data-dependent and drifts during training.

**Q2. The LSTM highway path is diagonal and bounded. Why is that still not enough?**
- **Answer:** Because the diagonal path is not the *only* path. Expanding `∂L/∂C_k` in the LSTM gives a term `∂L/∂h_t · ∂h_t/∂C_t · Π diag(f_j)` — the highway — but also terms in which the gradient re-enters through `h`, `o`, `i`, `g` and therefore passes through `W_·ᵀ` matrices with the same `σ'`-attenuated products the vanilla RNN suffers. Empirically, the highway term dominates only while `f̄`, the mean forget gate, stays near 1. Once the task requires frequent forgetting — which most tasks do, because irrelevant context must be discarded — `f̄` falls and the highway decays. The bound `Π f_j ≤ 1` is only useful if `f` stays close to 1, and the loss gives the network no explicit pressure to keep it there.
- **Why the interviewer asks this:** It is the difference between reciting "gradient highway" and understanding why LSTMs still needed TBPTT.
- **Trap:** Assuming the gates create a *protected* channel. They create a *gated* channel; protection would require `f = 1` unconditionally, which forfeits forgetting.

**Q3. Why do LSTM implementations initialise the forget-gate bias to 1?**
- **Answer:** `b_f = 1.0` makes `f_t = σ(1) ≈ 0.731` at initialisation instead of `σ(0) = 0.5`, so the cell state starts closer to "remember everything". This matters because at init the cell contents are near zero and the forget gate's job is undefined; a 0.5 forget gate halves the state every step, so any signal must be re-learned each step and the effective horizon at init is a handful of steps. The bias shifts the model into the remembering regime so gradient descent can explore forgetting from there. It is one of the highest-leverage one-line changes in the entire RNN literature (Gers & Schmidhuber 2000).
- **Why the interviewer asks this:** It is a small detail with a big effect that only people who have implemented an LSTM from scratch know.
- **Trap:** "It prevents vanishing gradients." It delays the onset of forgetting at initialisation; training can and does drive `b_f` back down.

**Q4. Explain the fixed-size hidden state as an information bottleneck.**
- **Answer:** The state carries at most `d_h` real numbers, quantised to the precision of the representation. At `d_h = 256` in fp32 that is 1024 bytes. A 1000-token input with 16-bit token entropy carries roughly `1000 × 16 bits = 2000 bytes` of information, more than the state can hold. So the encoder is a lossy compressor with no learned rate control: it must decide, before seeing the task, what to throw away. There is also a *distance* effect: information written at step 10 must survive 990 overwrites by an additive recurrence whose Jacobian norm is `ρ` per step, so the signal-to-noise ratio of that content decays as `ρ^990`. Attention removes the compression entirely — the representation of the past is `n` vectors, i.e. `n·d_model` numbers with a direct lookup path — which is why long-context work is possible at all.
- **Why the interviewer asks this:** Distinguishes "gradients don't flow" (an optimisation claim) from "the information isn't there" (a capacity claim). They are independent failure modes.
- **Trap:** Treating it as a purely gradient problem. Even with a perfect gradient oracle, a 256-float state cannot faithfully carry a 1000-token document.

**Q5. Name the three distinct reasons long-range dependencies break.**
- **Answer:**
  1. **Gradient failure** — the credit-assignment signal decays as `(γρ)^{t−k}`; the loss *cannot* influence distant parameters. Fixed by architectural gradient paths (residuals, LSTM highway, attention's O(1) path).
  2. **Information failure** — the fixed-size state cannot *represent* the distant content even if the gradient flows perfectly.
  3. **Optimisation failure** — the model has no *incentive*. If the task's labels are predictable from local context alone (short reviews, bag-of-words sentiment), the empirical risk is minimised by ignoring the distant tokens, and gradient descent will find that solution first because it is easier. You can have a working gradient path and full capacity and still get a model that never uses long context — this is the failure mode that survives into the transformer era and is why long-context benchmarks are full of "the model attends locally" results.
- **Why the interviewer asks this:** This is the highest-signal question in the module. Candidates who collapse all three into "vanishing gradients" get marked as having read a blog post.
- **Trap:** Assuming fixing (1) fixes (3). Attention fixes the gradient path completely, and models still fail to use 100 k-token context ("lost in the middle").

**Q6. Why does the softmax's row-stochasticity matter?**
- **Answer:** Three consequences. (1) The output is a **convex combination of the value vectors**, so `‖attn_out‖ ≤ max_j ‖v_j‖` — attention cannot amplify, only mix. (2) The weights sum to 1, so attention has no notion of "attend to nothing"; even when no key is relevant the layer must return a full mixture, which is a real limitation and the motivation for "attention sink" findings (models dump probability mass on the first token when they want to ignore the context). (3) Softmax is shift-invariant, so only score *differences* matter — which is why the `√d_k` scaling changes behaviour (it changes differences) while adding a constant to all scores would not.
- **Why the interviewer asks this:** It tests whether you reason about attention as a mathematical object rather than a diagram.
- **Trap:** "Attention weights sum to 1, so they're interpretable probabilities." Summing to 1 is necessary for interpretability but nowhere near sufficient — see the next question.

**Q7. Are attention weights explanations?**
- **Answer:** No, and there is direct experimental evidence (Jain & Wallace 2019; Serrano & Smith 2019). Attention weights are *one* intermediate quantity in a computation with residual connections, LayerNorm, multiple heads, and an FFN; the layer's contribution is `Σ_j α_j v_j W_O`, so a large `α_j` on a value whose `W_O v_j` is near zero contributes nothing. Perturbing the highest-weight positions often changes the output far less than perturbing low-weight ones. What is defensible: attention weights as a *debugging* signal for gross failures (all mass on `[PAD]`, all mass on the first token, uniform across heads) — and as a hypothesis generator to be tested by ablation.
- **Why the interviewer asks this:** It is the standard "do you overclaim" question, common in applied-research and product-facing loops.
- **Trap:** "Yes, attention tells you what the model looked at." It tells you what the *softmax* looked at, in one head, in one layer, before the value projection and the residual add.

**Q8. Why is attention's output low-rank in practice, and what are the consequences?**
- **Answer:** Each head produces `Σ_j α_j v_j` with the `α` row-stochastic, so the head's output lies in the convex hull of the `n` value vectors — a set of dimension at most `n − 1` (and at most `d_head`). Concatenating `h` heads gives at most `h · min(n, d_head)` effective dimensions, but the *effective* rank is empirically much lower: attention matrices are strongly rank-deficient, with a few dominant singular values and a long tail, and several heads are nearly redundant. Consequences: (a) attention layers can often be pruned or merged with small loss; (b) low-rank approximations of attention (Linformer-style projections) work because they are exploiting a property the layer already has; (c) the FFN is what restores rank and capacity, which is part of why it holds two thirds of the parameters.
- **Why the interviewer asks this:** Tests linear-algebra intuition about the layer, and it is a standard research-tier probe. `[Company style: big-tech research]`
- **Trap:** "Low-rank means you can always replace attention with a linear layer." Only for a fixed context length and only with task-specific loss of quality; the rank is data-dependent and grows with `n`.

**Q9. What did the `Attention Is All You Need` ablation table show?**
- **Answer:** Table 3 (variations on the base model) reports BLEU on WMT 2014 EN-DE. Key rows: the base transformer is **25.8**; single-head attention drops to **24.9** (−0.9); removing positional encodings entirely drops to **25.7 / 25.3** depending on variant; reducing `d_k` (which removes the need for the `1/√d_k` scaling) drops to **25.1**; and replacing scaled dot-product with additive attention is roughly neutral. The two conclusions interviewers are listening for: (1) **multi-head matters but modestly** — 0.9 BLEU, not a revolution; (2) **the big win is architectural, not the attention variant** — the model's advantage comes from removing recurrence and enabling parallelism across the whole sequence, not from any particular scoring function.
- **Why the interviewer asks this:** It measures whether you have read the paper or its summary, and it is a nice corrective to over-claiming about multi-head.
- **Trap:** "The paper proved attention alone is better than recurrence." The paper showed a *specific* architecture trained in 12 hours on 8 GPUs beat the previous best BLEU — a claim about training efficiency and quality *together*. The "LSTMs are obsolete" reading is a later, stronger claim.

**Q10. The transformer and the LSTM trained on comparable data — why did the transformer win so decisively?**
- **Answer:** Not primarily because attention is a better function approximator; on small data an LSTM is often competitive. The win is a *scaling* win on three coupled axes. (1) **Parallelism:** the transformer's forward pass is GEMMs, so it saturates tensor cores (~50 % of peak vs ~0.6 % for a sequential RNN), giving roughly two orders of magnitude more throughput; (2) that throughput makes **larger corpora and larger models** affordable, and the empirical scaling laws (Kaplan 2020; Hoffmann 2022) are what convert compute into quality; (3) **gradient path length** is O(1) between any two positions, so the model can actually *use* the extra capacity on long-range structure rather than just having it. Remove any one of the three and the story weakens — which is exactly why hybrids (attention + linear recurrence) are viable now: they keep (3) while buying back (1)'s cost at long `n`.
- **Why the interviewer asks this:** It tests whether you think in terms of compute budgets and scaling laws rather than architecture aesthetics.
- **Trap:** "Attention models long-range dependencies better, so they win." On a 128-token classification task the LSTM may win outright. The transformer's advantage is conditional on scale.

**Q11. Why does Pre-LN remove the need for a 4000-step warmup?**
- **Answer:** In Post-LN the block computes `x_{l+1} = LN(x_l + F(x_l))`. The gradient through the residual stack is multiplied by the LayerNorm Jacobian at every layer, and LN's Jacobian has `1/σ` in it where `σ` is the token's activation standard deviation — small at init, so `1/σ` is large. Composed over `L` layers, the effective gradient magnitude depends strongly on depth and initial variance, and early in training `σ` is drifting. High learning rates then produce diverging updates, so the original recipe warmed up over 4000 steps to let `σ` settle. Pre-LN computes `x_{l+1} = x_l + F(LN(x_l))`, so the identity path is exact and unscaled: `∂x_{l+1}/∂x_l = I + ∂F/∂x`, and depth no longer multiplies a data-dependent factor into the gradient. Modern practice: Pre-LN or RMSNorm, warmup over 100–2000 steps still used for stability but no longer a hard requirement.
- **Why the interviewer asks this:** It is the deepest "internals" question in the block, and it connects normalisation placement to optimisation dynamics.
- **Trap:** "Pre-LN normalises the residual, so no warmup." It normalises the *branch input*, deliberately leaving the residual stream unnormalised — that is the entire point.

**Q12. FlashAttention does not reduce FLOPs. Why is it much faster?**
- **Answer:** Because standard attention is **memory-bound**, not compute-bound. Materialising the `n × n` score matrix means writing `n²` elements to HBM and reading them back for the softmax and the `·V` multiply — three round trips over `O(n²)` data. At `n = 4096` and 32 heads that is `4096² × 32 × 2 B = 1.07 GB` per layer, per forward pass, plus the same again for the backward. FlashAttention tiles `Q`, `K`, `V` into blocks that fit in on-chip SRAM (A100: 192 KB per SM), computes the softmax blockwise using the **online softmax** running-rescale identity — maintaining a running max `m` and running denominator `ℓ`, rescaling previous partial results when a new block raises the max — and never writes the full score matrix. HBM traffic drops from `O(n²)` to `O(n²/M)` for block size `M`, and the measured speedup is 2–4× forward, up to 2–3× more in the backward pass, with bit-level-equal results (up to floating-point reassociation).
- **Why the interviewer asks this:** It is the canonical "you must know the memory hierarchy" question, and it corrects the misconception that progress in attention has been about FLOPs.
- **Trap:** "FlashAttention approximates attention." It is exact. Also do not confuse it with sparse or linear attention, which *do* change the arithmetic.

**Q13. What is the arithmetic intensity of attention, and when does it cross the roofline?**
- **Answer:** Attention's HBM traffic per layer is `Q, K, V` reads plus output writes: `4 n d_model · 2 B` (fp16) ≈ `8 n d_model` bytes; its FLOPs are `≈ 4 n² d_model`, so intensity `≈ n/2` FLOP/byte. On an A100 needing `≈ 156` FLOP/byte to saturate, attention only becomes compute-bound around `n ≈ 300`; below that it is bandwidth-bound, which is exactly why FlashAttention helps most at short-to-medium `n` and why the whole kernel is designed around SRAM residency. The FFN, by contrast, is a GEMM with intensity `≈ n`, so it is compute-bound for `n ≳ 156` and typically runs at high MFU. Practical implication: when profiling a transformer at short context, the FFN and the LayerNorms dominate wall-clock, not attention — the opposite of the naive FLOPs picture.
- **Why the interviewer asks this:** It is the roofline model applied to a real kernel, which is a senior-infra signal. `[Company style: big-tech research]`
- **Trap:** Estimating speedup from FLOP count. If a change removes 50 % of FLOPs from a bandwidth-bound kernel, it may deliver ~0 % wall-clock improvement.

**Q14. Mamba makes `B`, `C` and `Δ` input-dependent. What does that buy, and what does it cost?**
- **Answer:** In an LTI SSM, `B`, `C`, `Δ` are fixed, so the recurrence is a fixed convolution with a fixed kernel — it can only *filter* by position, not by content. Making them functions of the input makes the system **selective**: it can choose to let a token's information into the state (large `Δ`), to hold it, or to gate it out (small `Δ`), which is precisely the copy/recall behaviour LSTMs get from their gates. The cost is that time-variance destroys the convolution view: with input-dependent parameters the system is no longer LTI, so you can no longer evaluate training as a single FFT convolution. Mamba's answer is a **hardware-aware parallel scan** — the selective recurrence is a first-order linear recurrence, which is associative, so a Blelloch-style scan computes all `n` states in `O(n log n)` work with `O(log n)` depth, and the kernel keeps the state in SRAM to avoid the materialised `(B, L, D, N)` intermediates.
- **Why the interviewer asks this:** It is the technical core of the main attention alternative and tests whether you understand *why* selectivity was the contribution.
- **Trap:** "Mamba is just an RNN with a fancy kernel." The selectivity is the modelling contribution and the scan is the engineering contribution; the LTI variant (S4) already existed with the convolution trick.

**Q15. When do SSMs still lose to attention?**
- **Answer:** On **recall** and **associative** tasks. A fixed-size state must overwrite as it goes, so retrieving an arbitrary token from `k` steps back requires the information to have survived every intervening write — an SSM has no addressing mechanism, only content-based filtering. Concretely: MQAR (multi-query associative recall) and induction-head tasks show transformers winning by wide margins at fixed parameter count; copying a random string from the middle of a long context is the canonical failure. There is a theoretical argument that a fixed-size recurrent state cannot implement certain recall functions at all regardless of training (the state-capacity bound), which is why the empirical fix is not "train the SSM harder" but "add attention layers" — every strong hybrid puts attention exactly where recall-heavy patterns need it.
- **Why the interviewer asks this:** It is the balanced view that separates someone who read the Mamba abstract from someone who can choose an architecture.
- **Trap:** "SSMs are strictly worse." They match or beat transformers on long-context throughput-bound tasks and on sequence modelling where the signal is smooth (audio, time series, genomics), and they have no KV cache at inference.

**Q16. Explain the path-length comparison: O(1) vs O(log n) vs O(n).**
- **Answer:** Path length is the number of sequential operations a signal must traverse between two positions that need to interact — it bounds both gradient propagation and information mixing.
  - **RNN/LSTM: `O(n)`** — interaction between positions 1 and `n` requires `n` serial steps, and the gradient travels a product of `n` Jacobians.
  - **Transformer: `O(1)`** — every position attends to every other in one matrix multiply; the path from token `i` to token `j` is one attention hop, and the residual stream adds a direct `I` path across the whole depth.
  - **SSM/Mamba: `O(log n)`** on hardware (the parallel scan's depth), but the *information* path is still a linear recurrence whose state must be overwritten — so the practical recall path behaves like `O(n)` even though the compute depth is logarithmic. This distinction between *computational* depth and *representational* path is the subtlety interviewers probe for.
  - **Log-depth hybrids / hierarchical models: `O(log n)`** by chunking — which is exactly what block-sparse and hierarchical attention do.
- **Why the interviewer asks this:** It is the standard comparative framework and it is regularly mis-answered by conflating scan depth with information path.
- **Trap:** Quoting `O(log n)` for Mamba without the caveat. The scan is log-depth in *parallel time*; it does not give the state log-depth access to the past.

**Q17. What is the residual stream, and why does it matter for parameter distribution?**
- **Answer:** The residual stream is the `d_model`-wide vector that every block reads from and writes to by addition. Because blocks *add* rather than replace, `d_model` is a shared communication bus of fixed width across all `L` layers, and the model's total "working memory" is `L × d_model` writes superimposed in one `d_model`-dimensional space. Two consequences. (1) **Superposition:** the stream must represent far more features than it has dimensions, so features are stored as near-orthogonal directions and read out by linear projections — this is the mechanistic-interpretability picture and it explains why individual neuron interpretations are unreliable. (2) **Parameter placement:** since the stream is width-limited, capability must be bought with depth and with the FFN's width; the FFN's `4d` expansion gives each layer a private high-dimensional scratch space (`d_ff` wide) before projecting back down, which is where computation like key-value lookup happens. It is also why LoRA targets attention and FFN projections rather than LayerNorms — the residual stream itself has no parameters.
- **Why the interviewer asks this:** It is a modern-interpretability framing that also justifies practical PEFT choices (which matrices get adapters).
- **Trap:** "The residual stream is just a gradient aid." It is also the model's shared representational workspace, and that is the more consequential role.

**Q18. Why does gradient clipping interact badly with adaptive optimizers at the edge of stability?**
- **Answer:** Clipping rescales the global gradient by `clip/‖g‖`. Adam then normalises by the *running* RMS of the pre-clip gradient magnitudes. If clipping fires on most steps, the gradient Adam sees has an artificially compressed dynamic range, so its `v` estimate understates the true variance — and when clipping does *not* fire, the update is comparatively enormous. The combination is worst on RNNs, where the gradient norm distribution is heavy-tailed: clipping fires rarely but on the exact steps carrying the long-range signal, so clipping systematically discards the informative gradients and keeps the local ones, which is the opposite of what you want. Practical mitigations: clip by norm at a threshold tuned by watching the *fraction* of clipped steps (target < 10 %), or clip in `torch` before passing to the optimizer and monitor `clip_coef` from `clip_grad_norm_`'s return value.
- **Why the interviewer asks this:** A senior-level interaction bug that only shows up when you instrument clipping rather than just enabling it.
- **Trap:** "Clipping is free insurance." It has a systematic bias, and on heavy-tailed RNN gradients the bias removes the signal you care about. `clip_grad_norm_` returns the coefficient — log it.

**Q19. Why is "attention is all you need" not a claim about FLOP efficiency?**
- **Answer:** The paper's argument is about **path length and parallelisability**, not operation count. Per token, a self-attention layer does `≈ 4 n d_model` FLOPs while a recurrent layer does `≈ 4 d_model²` — for `n < d_model` (e.g. `n = 512`, `d_model = 4096`) the recurrent layer is *cheaper* in FLOPs per token, and by a wide margin. The transformer wins because those FLOPs are laid out as dense GEMMs that the hardware can run at ~50 % MFU instead of ~0.6 %, because the O(1) path length makes the capacity usable, and because the parallelism allows the model to be scaled at all. Any claim of the form "attention is cheaper" is wrong in general and right only at the wall-clock/utilisation level at practical lengths.
- **Why the interviewer asks this:** It is the direct test of whether you understood the previous questions or just filed them as slogans. `[Company style: big-tech research]`
- **Trap:** "Attention is O(n²) so it's always worse asymptotically at long context." Crossover matters: below `n ≈ 300–1000`, attention is bandwidth-bound and often *faster in wall-clock* than a linear-recurrence kernel with poor hardware utilisation.

**Q20. Why does the LSTM output gate matter for the gradient into `C`?**
- **Answer:** `h_t = o_t ⊙ tanh(C_t)`, so `∂h_t/∂C_t = o_t ⊙ (1 − tanh²(C_t))`. The output gate therefore sits directly between the cell state and everything downstream, including the loss. A small `o_t` — which is exactly what the network learns for "this step's state is not currently relevant" — blocks the *incoming* gradient to `C_t` even though the highway path backwards from `C_t` is healthy. This is a real and under-discussed LSTM failure: the highway is only useful if gradients can get *onto* it, and the output gate controls that on-ramp, while the input gate `i_t` and candidate `g_t` control the other on-ramp (`∂C_t/∂g_t = i_t`). So all three of `f` (through-path), `i` and `o` (on-ramps) must be simultaneously favourable for long-range learning.
- **Why the interviewer asks this:** It tests whether the "highway" picture is complete. It usually is not.
- **Trap:** Treating `o_t` as only an output read-out gate. It gates the gradient's entry to the cell, not just the value's exit.

**Q21. Why can't you just increase `d_h` to fix the bottleneck?**
- **Answer:** Three costs scale badly. (1) **Parameters and compute** grow as `d_h²` in `W_hh`, so `d_h: 256 → 4096` is a 256× increase in recurrent parameters and FLOPs per step, and those FLOPs are a matvec at ~1 FLOP/byte, so wall-clock grows linearly while MFU stays near the floor. (2) **Optimisation** gets harder: a larger `W_hh` has a spectral radius closer to the `γρ = 1` knife edge and its gradient product is correspondingly more unstable — the vanishing problem gets *worse*, not better, at the same time as the capacity problem gets better. (3) **The compression is still lossy at fixed `n`.** Larger `d_h` helps only until the state is big enough to hold the sequence; for a 100 k-token document even `d_h = 10^5` is not a faithful store, and you have to pay the quadratic price anyway. This is the argument for changing the *architecture* rather than the hyperparameter: attention is a memory of size `n·d_model` with a learned addressing scheme, at `O(n)` memory instead of `O(d_h²)`.
- **Why the interviewer asks this:** It is the "why not just scale the obvious knob" question, and the honest answer requires all three axes.
- **Trap:** Answering only about parameters/compute. The optimisation-difficulty point is the one that actually kills the idea.

**Q22. ULMFiT got transfer learning to work on an LSTM in 2018. What does that tell you about the "transformers enabled fine-tuning" narrative?**
- **Answer:** That the narrative is about *economics*, not possibility. ULMFiT showed that with careful training (discriminative learning rates, slanted triangular schedules, gradual unfreezing from the top down) a 3-layer AWD-LSTM could transfer from a general corpus to a target task and beat from-scratch models trained on 100× more data, with 100 labelled examples — 18–24 % error reduction on six datasets. What transformers changed was not that transfer became possible but that it became *cheap and automatic*: attention removed the architecture/label-space coupling, made a single generic pretraining objective produce representations that survive to arbitrary tasks, and made the compute affordable at scale. The engineering point for a practitioner: the fine-tuning recipe (unfreeze progressively, small LR on early layers, larger on the head) is an LSTM-era invention that survives verbatim in today's LoRA learning-rate schedules.
- **Why the interviewer asks this:** It rewards knowing the actual history and it separates "fine-tuning is a thing we do" from "here is why the recipe looks the way it does".
- **Trap:** "ULMFiT was superseded so it has no lessons." Its schedule and unfreezing ideas are in every modern fine-tuning script.

---

## Level 4 — System Design & Scenario

Answer these in five beats: **requirements → constraints → design → trade-offs → failure modes.** Interviewers grade the structure as much as the answer.

**Q1. Design a sentiment classifier over 10 M customer reviews. Do it for 2018 and for 2026, and say what changed.**
- **Answer structure:**
  - **Requirements.** Binary or 5-class sentiment, ~10 M labelled reviews, p99 latency < 200 ms, retrain monthly on new data, per-language variants, must not regress on historical slices.
  - **Constraints.** 2018: GPUs are `V100`-class, no pretrained general-purpose text encoder in wide use yet (ELMo is 3 months old and slow — an LSTM over 10⁷ examples per inference pass is not servable at 200 ms without batching and truncation). 2026: pretrained 7–8 B decoder or 110 M–340 M encoder available, and LoRA/QLoRA adapters make per-task training a single-GPU job.
  - **Design 2018.** Tokenizer learned on the review corpus → embedding 128 → 1–2 layer BiLSTM with `d_h = 256` → mean-pool over non-padded timesteps → dropout → linear. Train with TBPTT at `k = 200`, clip at 1.0, `padding='pre'` or masked. Budget: a BiLSTM at `d_h = 256` runs ~200 sequential steps per example; on a `V100` that is ~5 ms/example, so throughput is ~200 examples/s/GPU, i.e. ~14 GPU-hours per epoch over 10 M examples. Fine-tuning means retraining the whole model per language.
  - **Design 2026.** Frozen 7–8 B decoder (or a 340 M encoder for latency), LoRA rank 16 on `q_proj, k_proj, v_proj, o_proj`, 4-bit QLoRA, one epoch, effective batch 64, cosine schedule, `lr = 2e-4`. Train cost: ~0.5–2 GPU-hours on a single A100 for 10 M examples. Adapters are 20–50 MB each, so per-language variants are cheap and hot-swappable.
  - **Trade-offs.** The 2026 design wins on accuracy per labelled example, on training cost, and on multi-task flexibility (adapters). It loses on inference cost — a 7 B forward pass is far more expensive per example than a 5 M-parameter BiLSTM — so at very high QPS with tight latency, distil to a 110 M encoder or use a small linear head on frozen embeddings.
  - **Failure modes.** 2018: class imbalance invisible in accuracy (the notebook's `0.5011`), truncation past 200 tokens, vocabulary drift when re-tokenising new data, and no baseline-loss monitoring. 2026: LoRA rank too high overfits the minority class, quantisation degrades calibration (important if you threshold on probability), and prompt-template drift between train and serve.
- **Why the interviewer asks this:** Open-ended, and it grades whether you decompose by epoch and constraint rather than reciting a favourite architecture. `[Company style: consulting]`

**Q2. Design a summarizer for 100 k-token legal documents.**
- **Answer structure:**
  - **Requirements.** Faithful abstractive summaries of 100 k-token contracts; citations back to source spans (auditable); batch throughput matters more than latency; must handle tables.
  - **Constraints.** 100 k tokens does not fit any off-the-shelf 8 k model; attention is `O(n²)` so a 100 k context at 32 layers costs `4 × (10⁵)² × 4096 ≈ 1.6 × 10¹⁴` FLOPs per layer-pass — about 30 s–2 min per document on one A100 depending on kernel quality. Memory: KV cache at 128 KB/token × 100 k = 12.8 GB, which fits an 80 GB card with a small batch, or needs GQA/quantised cache.
  - **Design (two viable paths).**
    1. **Long-context transformer with RoPE interpolation** (YaRN / NTK-scaled), FlashAttention-2, paged KV cache, chunked prefill. One pass, highest faithfulness, needs the model to have been continued-pretrained at the target length — otherwise "lost in the middle" degrades recall of the contract's centre.
    2. **Hierarchical retrieve-then-summarise.** Chunk to 4 k tokens → embed → retrieve the top-k clauses per section with a structure-aware index (clause headings as keys) → summarise section by section → summarise the summaries. Cheaper, better citation control, and auditable; the risk is that cross-clause dependencies (a definition in §2 governing §14) are lost.
  - **Trade-offs.** Path 1 is 10–50× more expensive per document at 100 k and its quality is bounded by the model's effective context, which is typically well below its nominal context. Path 2 is cheap and interpretable but has a hard recall ceiling. Production answer: path 2 with path 1 for the final synthesis on the retrieved set — this is what most legal-AI systems actually ship.
  - **Failure modes.** Silent omission of a clause (unmeasurable by ROUGE); hallucinated dollar amounts; citation spans that point to the wrong chunk after re-chunking; and position bias — evaluate recall at 5 depths (0 %, 25 %, 50 %, 75 %, 100 %) rather than on average, because the average hides the middle.
- **Why the interviewer asks this:** Tests whether you know the difference between nominal context and effective context, which is the most common production miscalculation. `[Company style: big-tech applied]`

**Q3. You must serve a 100 k-context model at 50 requests/second. Design the serving stack.**
- **Answer structure:**
  - **Requirements.** 50 req/s at 100 k-token prefill, streaming output, ~1 k output tokens, p95 time-to-first-token < 5 s, cost per request known.
  - **Constraints.** KV cache dominates. Llama-3-70B with GQA (`n_layers = 80`, `n_kv_heads = 8`, `d_head = 128`, fp16) = `2 × 80 × 8 × 128 × 2 = 327 680 B` = 320 KB/token → 32 GB per 100 k sequence. One 80 GB H100 holds at most two concurrent 100 k sequences plus weights (140 GB in fp16 — does not fit at all), so 70 B fp16 is off the table; you need 4-bit weights (35 GB) plus 32 GB of cache per sequence, i.e. one sequence per card, and 50 req/s is impossible without aggregation.
  - **Design.** (1) **Quantise weights** to 4-bit AWQ/GPTQ or fp8 — halves to quarters the weight footprint. (2) **Quantise the KV cache** to fp8 (≈2× more sequences per card) — this is where the real headroom is. (3) **Chunked prefill + continuous batching** with a paged cache (vLLM `PagedAttention`, TensorRT-LLM, SGLang RadixAttention) so 100 k prefills interleave with decodes instead of blocking them. (4) **Prefix caching** if prompts share a long system/document prefix — for a fixed instruction preamble this can cut prefill cost by 90 %+. (5) **Scale horizontally** with a router; 50 req/s at 100 k is a fleet, not a card. (6) **Admission control / queueing** with a separate priority lane for short requests, because a 100 k prefill occupies the GPU for seconds.
  - **Trade-offs.** Cache quantisation costs a small amount of quality and must be validated; prefix caching only helps with shared prefixes; chunked prefill trades TTFT for throughput; disaggregated prefill/decode (separate GPU pools) maximises both but doubles operational complexity.
  - **Failure modes.** VRAM fragmentation (the reason paged caches exist); cache eviction storms under bursty load; TTFT collapse when a long prefill is admitted mid-batch; and per-request cost that silently 10×s when average context grows from 10 k to 100 k while the pricing model assumed the former.
- **Why the interviewer asks this:** Serving interviews live here, and the numbers (320 KB/token for 70 B, 128 KB/token for 8 B) are the currency of the conversation. `[Company style: big-tech infra]`

**Q4. Design a real-time streaming pipeline: transcribe a live call and flag compliance violations within 500 ms.**
- **Answer structure:**
  - **Requirements.** Audio in, transcript plus a violation flag out, < 500 ms added latency, high recall on violations (missing a violation is worse than a false alarm), full audit trail.
  - **Constraints.** Causality is the binding constraint: the model may only see tokens up to now, so bidirectional encoders (BERT) and full-attention over the whole document are unavailable. This is the one setting where an RNN's inductive bias is genuinely *right* — a causal, constant-memory, `O(1)`-per-step model with no growing cache.
  - **Design.** Streaming ASR with a chunked/emitting encoder → a causal classifier. Two options. (a) **Streaming transformer decoder** with a sliding window (e.g. 4 k tokens with a 512-token stride), which bounds the KV cache and gives strong in-context recall of the compliance rules. (b) **Causal SSM or small causal LSTM/GRU** over the token stream plus rule/regex detection for the unambiguous patterns — constant memory per stream, trivially parallel across thousands of concurrent calls, and it never needs a KV cache. In practice: **regex/lexicon first, small causal model second, transformer only for the final adjudication on a sliding window.** A two-stage design is what meets the latency budget.
  - **Trade-offs.** (a) has better accuracy and can hold the rule set in context; (b) is 10–100× cheaper per stream and has no context limit, but its recall on rare phrasings is weaker. The hybrid routes ~95 % of traffic to the cheap path.
  - **Failure modes.** ASR chunk boundaries splitting a violation phrase (mitigate with overlap); latency creep from queueing rather than compute; drift in the rule set that the model was not retrained on; and no replayable log, which makes post-hoc audits impossible.
- **Why the interviewer asks this:** It forces the candidate to say something positive about recurrence, which candidates trained only on the "transformers won" narrative cannot do. `[Company style: startup ML eng]`

**Q5. Design the fine-tuning pipeline for a 70 B model where every RNN-era blocker still applies in spirit.**
- **Answer structure:**
  - **Requirements.** Adapt a 70 B base model to a domain task with 5 k examples, on a fixed budget, with reproducible runs and a rollback path.
  - **Constraints.** Full fine-tuning is `70 × 10⁹ × 4` bytes of weights + grads + Adam moments `= 70 GB × (2 + 2 + 4 + 4) = 840 GB`, i.e. ≥ 11× 80 GB cards with ZeRO-3/FSDP. 5 k examples will overfit a full fine-tune badly. The "RNN-era" blockers map directly: **no reusable checkpoint at the right granularity** (a base model is not a task model), **task-specific output head**, **no tokenizer reuse** if you change the domain vocabulary, **tiny dataset**, and **per-task retraining cost**.
  - **Design.** QLoRA: 4-bit NF4 base with double quantisation (~35 GB, one A100-80GB), LoRA rank 32–64 on all attention and FFN projections, `alpha = 2 × rank`, dropout 0.05, `lr = 2e-4` with cosine decay, 2–3 epochs, effective batch 32–64, gradient checkpointing, paged optimizer. Held-out 10 % with a fixed seed; evaluate exact-match plus a task-specific metric plus a regression set from the base model's general ability.
  - **Trade-offs.** QLoRA is ~2× slower per step than 16-bit LoRA but fits on one GPU; LoRA quality at rank 64 approaches full FT on most classification/extraction tasks but lags on tasks requiring new *knowledge* rather than new *behaviour* (that is continued pretraining, not SFT — see CS-12). Rank too high overfits 5 k examples; rank too low underfits the domain shift.
  - **Failure modes.** Overfitting invisible in loss because the eval set leaked into training; adapter merged into a quantised base (numerically wrong — merge into the 16-bit base, then re-quantise); tokenizer unchanged but prompt template changed between train and serve; and no versioned adapter registry, so rollback means retraining.
- **Why the interviewer asks this:** It is the module's practical payload — the blockers changed shape but the discipline (evaluate against a baseline, version the artifact, plan the rollback) did not.

**Q6. Choose an architecture for a 1 M-token log anomaly detector: transformer, hybrid, or pure SSM?**
- **Answer structure:**
  - **Requirements.** 1 M tokens per window, streaming ingestion, detection latency under a minute, high recall on rare anomalies, cost budget per GB of logs.
  - **Constraints.** Pure attention at 1 M tokens is `4 × (10⁶)² × d = 4 × 10¹² × d` FLOPs per layer-pass — at `d = 4096` that is `1.6 × 10¹⁶` FLOPs, hours on a single GPU. KV cache is prohibitive (128 KB/token × 10⁶ = 128 GB for an 8 B model). So full attention is out; the choice is *how* to avoid it.
  - **Design.** Pure SSM (e.g. a Mamba-2 stack) if the signal is aggregate/statistical — throughput anomalies, error-rate drift, latency distribution shift — because these are *summarisation* patterns where a fixed state is genuinely sufficient, and the model has no KV cache so 1 M tokens streams at constant memory. **Hybrid** (attention every 6–8 layers, Mamba otherwise) if you need exact recall of specific identifiers: a request ID appearing 800 k tokens earlier, a stack trace that matches a template, a "seen this exact signature before" judgement. Anomaly detection in practice needs both, so default to the hybrid.
  - **Trade-offs.** Pure SSM: cheapest, best throughput, weakest recall. Hybrid: ~1.5–2× the SSM cost, recovers most recall, still linear in `n`. Full attention: best recall, unaffordable. Also consider the non-neural baseline first — for many log-anomaly tasks a frequency/template model (Drain-style parsing plus statistical tests) gets 80 % of the value at 1 % of the cost, and that comparison should be in the design.
  - **Failure modes.** Anomaly rate drift making the decision threshold stale; the model learning the *log format* rather than the anomaly (train on a held-out service to test generalisation); evaluation on a period with a known incident that the training window already contained (leakage); and silent degradation when the log schema changes upstream.
- **Why the interviewer asks this:** It is the question that makes the SSM discussion concrete, and the "what is the cheap baseline" answer is what distinguishes a senior design. `[Company style: big-tech infra]`

**Q7. Design an experiment that proves whether your model actually uses long-range context.**
- **Answer structure:**
  - **Requirements.** A falsifiable test, not a benchmark score. Must distinguish the three failure modes (gradient, information, optimisation) and must not be confounded by the model's short-range ability.
  - **Constraints.** Any test must hold the local context fixed while varying only the distant content, and must have a control in which the distant content is randomised.
  - **Design — four tests, in increasing strength.**
    1. **Position-controlled recall.** Place a critical fact at depth `d ∈ {0 %, 25 %, 50 %, 75 %, 100 %}` of a `n`-token context, ask a question answerable only from that fact, and plot accuracy against depth. A flat curve means context is used; a U-shape ("lost in the middle") means only the ends are.
    2. **Distance sweep at fixed local content.** Hold the last 500 tokens identical and move the single informative token from 500 to 100 000 tokens back. Any drop is a distance effect, isolated from local ability.
    3. **Ablation control.** Repeat (2) with the informative token replaced by a random token of the same length. The gap between the two curves is the model's *use* of the content; the control's above-chance accuracy is the model's guessing/format prior.
    4. **Gradient-flow instrumentation.** For the RNN-era question specifically, log `‖∂L/∂h_k‖` averaged over batches as a function of `t − k`. A geometric decay with the measured slope is direct evidence of the `(γρ)^{t−k}` bound, and it separates a gradient failure from an information failure — if the gradient reaches the distant step but accuracy still fails, the state cannot represent the content (information failure) or the model has no incentive (optimisation failure).
  - **Trade-offs.** Tests 1–3 measure behaviour, not mechanism; test 4 measures mechanism but only for a specific batch and layer. Report both. Also: use a synthetic task with a known answer key rather than a natural-language benchmark, so you are not measuring the benchmark's annotation noise.
  - **Failure modes.** Confounding the probe with the model's tokenizer (the "needle" tokenizes differently at different depths); measuring only one seed; and reporting mean accuracy over depths, which hides the U-shape entirely — report the curve.
- **Why the interviewer asks this:** It is the single best discriminator for "does this candidate evaluate rigorously". `[Company style: big-tech research]`

**Q8. Migrate a production LSTM classifier to a transformer with zero downtime.**
- **Answer structure:**
  - **Requirements.** Existing LSTM serving 5 k QPS with committed latency and calibrated probabilities; new transformer must not regress on any monitored slice; rollback within one minute at any point.
  - **Constraints.** Probabilities are calibrated differently (a fine-tuned transformer's softmax is typically *more* overconfident), downstream thresholds are tuned to the old distribution, and the LSTM's tokenizer/vocab must not be assumed compatible. There is a frozen API contract.
  - **Design — five phases.** (1) **Build the shadow.** Train the transformer on the same labels; add a temperature or Platt scaling step fitted on a held-out set so the output distribution matches the incumbent's. (2) **Offline parity.** Score 30 days of production traffic with both models, compare at the *decision* level (agreement rate, per-slice accuracy, calibration curves, and the disagreement set — read 100 disagreements by hand). (3) **Shadow serve.** Run both in production, log the new model's outputs, serve the old model's. Nothing changes for the user. Measure live agreement and latency. (4) **Canary.** Route 1 % → 5 % → 25 % → 50 % of traffic, with automated rollback on any slice regression beyond a pre-registered threshold. (5) **Cut over and keep the old model warm for one week** — a serving replica, not a checkpoint on disk, so rollback is a router flip.
  - **Trade-offs.** Shadow serving doubles inference cost for the migration window (budget for it). A softer ramp is slower but catches tokenizer/prompt drift that offline parity cannot see. Calibration matching costs a few points of raw accuracy in exchange for not breaking downstream thresholds — usually the right trade.
  - **Failure modes.** Vocabulary mismatch: if you switch tokenizers, the frozen downstream feature store (if any) becomes invalid. Latency: a 7 B transformer at 5 k QPS needs batching and probably distillation; the LSTM at 5 k QPS likely ran on CPU. And the classic — a canary that passes because the 1 % slice is unrepresentative of the heavy-tail traffic that arrives at peak.
- **Why the interviewer asks this:** It tests migration discipline, which is a staff-level skill, and it is a plausible real scenario given this module's content. `[Company style: big-tech applied]`

---

## Level 5 — Debugging & Incident Response

For each: state your **ordered** checks, and say what each one rules in or out. Order matters more than completeness.

**Q1. Loss is pinned at exactly `ln(vocab_size)` and has not moved in 2000 steps.**
- **Answer — ordered:**
  1. **Empirical label distribution.** Histogram the targets. Uniform → the labels are random or misaligned; nothing downstream matters. This is the notebook's bug and the single most common cause.
  2. **Label shift.** Verify `input = tokens[:, :-1]` and `target = tokens[:, 1:]`. An off-by-one produces a stationary loss near `ln(V)`.
  3. **Loss input type.** `nn.CrossEntropyLoss` expects **logits**. If you passed `softmax(logits)` the loss bottoms out high and flat. Also check `ignore_index` is set for padding, or the padded positions (which are a constant token) inflate and flatten the average.
  4. **Head liveness.** `print(logits.std(dim=-1).mean())` — if `~1e-6`, the output projection is dead (zero-init or a frozen layer).
  5. **Optimizer wiring.** Constructed after the model's parameters exist? `param.grad is not None` after `backward()`? Any `requires_grad=False` on the right prefix?
  - **Ruled out by (1):** all model-side hypotheses. **Ruled in by (3):** a loss-function bug that looks like a data bug.
- **Why the interviewer asks this:** It is the canonical L5 opener for this module and it has a precise signature — a *constant* loss at a value equal to a known constant is a data/wiring signature, not an optimisation signature.
- **Trap:** Starting with the learning rate or the architecture. Neither produces exactly `ln(V)` at step 0.

**Q2. Loss spikes to NaN at step ~300, always around the same step.**
- **Answer — ordered:**
  1. **Gradient norms over the last 100 steps.** If `‖g‖` rises geometrically then jumps to NaN, this is the exploding branch of the same product that causes vanishing. If `‖g‖` is flat and the *loss* spikes first, look at the data.
  2. **Which batch?** Log the offending step's input ids, lengths, and label. A single corrupt sample (all `-100` labels, an all-padding batch, a `NaN` in the embeddings) is a very common cause and shows up at a deterministic step.
  3. **Data scale.** Any unbounded input feature (unnormalised numeric columns, long sequences with no truncation) produces large activations and overflow in fp16. fp16 overflows above 65 504 — if you are in fp16 and any activation exceeds that, you get inf → NaN. Switch to bf16 (same exponent range as fp32) and re-run; if the NaNs vanish, it was overflow.
  4. **Loss reduction with zero-length targets.** `mean` over a batch where one example has 0 valid tokens gives a division by zero → NaN that then propagates into the weights and stays.
  5. **Learning rate / warmup.** If (1) shows a gradual rise, reduce the peak LR or lengthen warmup, and enable clipping at 1.0 — but only after ruling out (2)–(4), because clipping will just delay a data-driven NaN.
- **Why the interviewer asks this:** Determinism at a fixed step is the key clue, and it points at data or at a schedule; a non-deterministic NaN points at numerics.
- **Trap:** Immediately lowering the learning rate. It is the right move for one of the five causes and a waste of a training run for the other four. Also: `assert torch.isfinite(loss)` and `torch.autograd.set_detect_anomaly(True)` on a short repro run are the two tools to reach for first.

**Q3. Loss decreases to 0.69 on binary classification and stays there; accuracy is 0.51.**
- **Answer — ordered:**
  1. **Recognise the number.** `ln 2 = 0.6931`. A binary cross-entropy resting at `ln 2` means the model outputs the base rate — 0.5 for a balanced set — for every example. This is exactly the notebook's `0.6942 / 0.5011`. It is not a convergence problem; the model has learned nothing.
  2. **Class balance.** If the set is balanced at 50/50, `ln 2` is the trivial baseline. If it is 90/10, the trivial baseline is `−[0.9 ln 0.9 + 0.1 ln 0.1] = 0.325`, and a loss of 0.69 is *worse* than trivial — a different and more urgent bug.
  3. **Is the model getting the input?** Freeze everything and train only the head. If the loss still does not move, the features are constant — check for a `Masking`/pooling bug that averages over padding, a `padding`/`truncating` mismatch, or an incorrectly shaped input silently broadcast.
  4. **Steps, not epochs.** `steps = n_examples / batch_size × epochs`. The notebook's 47 steps (1 epoch, 3000 rows, batch 64) cannot train a 1.67 M-parameter model. Compare against a target of ≥ 500–2000 steps for a small task.
  5. **Truncation vs label.** If labels are only predictable from the *end* of a review and you truncate at 200 tokens with `padding='post'`, the label is genuinely unpredictable from the input. Verify by training on full-length inputs.
- **Why the interviewer asks this:** The reflex to convert a loss into a baseline comparison is the single most valuable habit in applied ML, and the specific value `ln 2` should be instant recall.
- **Trap:** Reading `0.5011` accuracy and reporting "the model is still learning".

**Q4. Gradient-norm plot shows `1e-7` across all layers from step 0, and never grows.**
- **Answer — ordered:**
  1. **Distinguish from "small but moving".** A constant `1e-7` is not vanishing dynamics — vanishing produces a *decaying profile with depth/step distance*. A uniform, constant small value is a scaling or wiring bug.
  2. **Loss scale and reduction.** If you are using `reduction='sum'` over a huge batch and dividing manually, or `mean` over a padded batch, the gradient is scaled by a factor `~1/N`. Compare against a run with `reduction='mean'` on a batch of 8.
  3. **Init scale.** A model initialised with an overly small `std` (e.g. `std=1e-3` instead of the framework default) produces small activations and correspondingly small gradients with no pathology. Check `logits.std()` at init.
  4. **Frozen/quantised parameters.** Under bf16 or with a `torch.no_grad()` accidentally wrapping part of the forward pass, gradients are not computed. Confirm the graph reaches the parameters: `loss.requires_grad is True` and `sum(p.numel() for p in model.parameters() if p.requires_grad)` is the number you expect.
  5. **Only then: the vanishing hypothesis.** If (2)–(4) are clean and the *relative* profile is decaying with distance — measure `‖∂L/∂h_k‖` against `t − k` — then it is genuine and the fixes are architectural (residual/gating/attention) not optimiser-side.
- **Why the interviewer asks this:** It punishes the reflex answer. "Vanishing gradients" is wrong for a constant profile, and interviewers use this question specifically to catch that reflex.
- **Trap:** Suggesting Adam, gradient clipping, or a larger learning rate before checking the scaling.

**Q5. The model works at 128 tokens and fails at 512, with the same weights.**
- **Answer — ordered:**
  1. **Positional encoding out of range.** Sinusoidal encodings are defined at all positions, but learned position embeddings have no row for index 512 and will either error or silently return `<unk>`. RoPE without scaling degrades sharply past the trained length. This is the most common cause by a wide margin.
  2. **Whether the model was trained at 512.** If training used `max_len = 128`, this is not a bug — it is distribution shift. Confirm the training config's `max_len`/`max_position_embeddings`.
  3. **Attention scaling or masking.** With a *learned* positional table, check `position_ids` are being passed (a common bug: the same position embedding for every position, or all zeros). With a causal mask, verify the mask has the right shape at 512 — a silently broadcast mask is a classic.
  4. **Truncation in the eval pipeline.** A tokenizer configured with `truncation=True, max_length=128` will cut your 512-token input and the model will be evaluated on its first 128 tokens.
  5. **For an RNN:** at 512 steps TBPTT at `k = 64` means the model has *never* had gradient information across a 512-step span, so failing at 512 is expected, not anomalous. This is the architectural statement of the module.
- **Why the interviewer asks this:** It is the most common real-world long-context incident and it has a clean, ordered diagnostic tree.
- **Trap:** Assuming it is a gradient problem. At inference there are no gradients — an evaluation-time failure is never a vanishing-gradient failure.

**Q6. Inference is 100× slower than the FLOP estimates predict.**
- **Answer — ordered:**
  1. **Compute the arithmetic intensity of your actual workload.** If you are running an RNN, or a transformer at batch 1, you are memory-bound at `≈ 1–2` FLOP/byte against a machine that needs ~150. This is not a bug; it is the roofline, and it explains the entire gap on its own.
  2. **Kernel launch overhead.** Serial per-step loops (an RNN step, a Python-level decode loop) launch a kernel per step per layer. At ~5 µs per launch and 200 steps × 2 layers = 400 launches = 2 ms of pure overhead per example, which can dwarf the arithmetic. Fix: fuse, use `torch.compile`, CUDA graphs, or batch the batch dimension up.
  3. **Materialised `n × n` attention.** Check peak memory: if it scales quadratically with `n` and you see the score matrix allocated, you do not have FlashAttention enabled. Verify with a profiler rather than assuming — `attn_implementation="flash_attention_2"` in HF, and `torch.backends.cuda.sdp_kernel` settings in raw PyTorch.
  4. **Sync points.** A `.item()`, `.cpu()`, `print(tensor)`, or metric update inside the loop forces a device synchronise every step. This is the single most common cause of a mysterious 10–100× slowdown in otherwise-correct code.
  5. **Precision and layout.** fp32 instead of bf16/fp16, or a non-contiguous tensor from a transpose that forces copies, or CPU offload silently enabled. Confirm dtype at every boundary.
- **Why the interviewer asks this:** It tests the memory hierarchy and profiler discipline rather than guessing. The right first move is *always* to profile before theorising.
- **Trap:** Blaming Python. It is more often arithmetic intensity or a sync point, and both are visible in a profiler trace.

**Q7. You fine-tune a model for 1 epoch and it gets worse at everything, including the target task.**
- **Answer — ordered:**
  1. **Effective learning rate.** LoRA at `lr = 2e-4` is fine; full fine-tuning at `2e-4` on a pretrained model is catastrophic forgetting. The correct range for full FT is `1e-5 – 5e-5`. Confirm which you are doing.
  2. **Warmup and schedule.** No warmup on a full FT destroys the pretrained weights in the first ~50 steps. Check `warmup_ratio ≥ 0.03` and a cosine/linear decay.
  3. **Steps.** `steps = N / batch × epochs`; 1 epoch over 5 k examples at batch 32 is 156 steps, which is usually too few to adapt but *plenty* to damage the model if the LR is wrong. Distinguish "too few steps to learn the task" from "enough steps to forget".
  4. **Regression evaluation.** Run the base model and the fine-tune on a held-out general-capability set. If general ability dropped and the task metric is flat, this is forgetting, not underfitting — different fix (lower LR, LoRA instead of full FT, replay a small fraction of pretraining data).
  5. **Masking and template.** Confirm the loss is computed only on completion tokens, not on the prompt (`ignore_index=-100` on the prompt span), and that the training template matches the evaluation template exactly — including the BOS token and whitespace.
- **Why the interviewer asks this:** It is the most common real fine-tuning incident, and it separates "lower the LR" from "add replay data" as answers.
- **Trap:** Assuming underfitting and training longer. If the model got *worse* on the target task, more steps at the same LR make it worse still.

**Q8. Validation loss is lower than training loss.**
- **Answer — ordered:**
  1. **Dropout / regularisation asymmetry.** Dropout is active in training and disabled in eval, so training loss is systematically higher. If the gap is ~1–3 % and both are decreasing, this is expected and requires no action.
  2. **The validation set is easier.** Check label balance, sequence-length distribution, and domain — a val split that is shorter or more templated than train will show lower loss forever. This is the most common *real* cause.
  3. **Training loss is a running average over a non-stationary window.** Keras/PyTorch report an epoch-mean over steps that were trained with a decaying LR; the val loss is computed once at the end with the final weights. For a fast-decaying schedule the two are not comparable. Fix by evaluating train loss on a fixed held-out subset at epoch end.
  4. **Leakage.** Validation examples appearing in training (duplicate rows, overlapping chunks in a sliding-window split, or a `validation_split` taken *after* augmentation). This is the one that matters — check for exact-duplicate overlap first.
  5. **Label noise in training only.** Mislabeled training rows raise train loss without affecting val.
- **Why the interviewer asks this:** It is the "know when not to act" question — a small, stable gap is not a bug, and a candidate who immediately proposes a fix overfits the metric instead of the model.
- **Trap:** "It means my model is fine." A *large and growing* inverse gap is leakage. Also note the notebook's own report: `validation_split=0.1` is 10 %, not the "0.1 %" the instructor says aloud.

**Q9. You load a saved Keras model and the predictions are garbage.**
- **Answer — ordered:**
  1. **Was the tokenizer saved with it?** Almost never. `model.save()` persists weights and architecture, not the word index. Reloading the data with a different `num_words` (the notebook does exactly this: 10000 → 1000) remaps every rare token to index 2 `<UNK>`, so the embedding lookup is fed indices that mean something different. Compare the first 20 token ids before and after.
  2. **Was the preprocessing identical?** `maxlen`, `padding`, `truncating`, and the `+3` index offset must match exactly. A `padding='pre'` model fed `padding='post'` input gives noise.
  3. **Dtype and shape.** `(1, 200)` int32 vs `(200,)`; float inputs where ints are expected. Keras will often silently accept and mis-broadcast.
  4. **Custom layers / legacy format.** HDF5 (`.h5`) is legacy — Keras 3 emits a deprecation warning and the file may lose custom objects or configuration. Migrate to `.keras` and re-export.
  5. **Was the model actually trained?** Sanity-check the baseline: for binary classification, does the model's output distribution differ from a constant 0.5? The notebook's loaded "classifier" outputs `0.4902` on the first review — a coin flip. The artifact was never a working model.
- **Why the interviewer asks this:** It is a direct read of the companion notebook's failure chain and it tests whether you version preprocessing alongside weights.
- **Trap:** Debugging the model. In the overwhelming majority of "reloaded model is garbage" incidents the model is fine and the preprocessing changed.

**Q10. During fine-tuning, training loss falls smoothly but the task metric stays at chance for the whole run.**
- **Answer — ordered:**
  1. **What is the loss actually computed on?** If you train with the language-modelling objective over the full sequence including the prompt, the loss falls by learning the prompt's distribution — which is trivial and unrelated to the task. Check that `ignore_index=-100` masks everything except the completion.
  2. **Compare the loss to the trivial baseline.** `ln(V)` for LM, `ln 2` for balanced binary. A loss of 2.1 against a `ln(32000) = 10.37` baseline looks like great progress — and it is, but if the *metric* is at chance the loss is measuring the wrong thing.
  3. **Evaluation mismatch.** Generation config (temperature, `max_new_tokens`, stop tokens, `pad_token_id`) can make a correctly-trained model score at chance on exact match. Print raw generations and read 20 by hand before touching the training.
  4. **Metric implementation.** String normalisation, whitespace, case, and the answer-extraction regex. A metric that fails to parse the model's output scores 0 regardless of quality.
  5. **Class/label imbalance or a constant-collapse solution.** If the model has learned to always emit the majority label, the metric is at the base rate while the loss is lower than `ln(V)`. Confusion matrix will show it immediately.
- **Why the interviewer asks this:** It is the "your metric and your loss are measuring different things" incident, which is extremely common in the SFT era and requires disciplined separation of the two.
- **Trap:** Lowering the learning rate or adding data. Neither helps when the loss and the metric are simply not measuring the same target.

---

## Rapid Fire — True / False / One-Liner

Answer these in under five seconds each. The verdict is what matters; the reason is what you say if asked to justify.

| # | Statement | Verdict + why |
|---|---|---|
| 1 | "The vanishing gradient problem means gradients become exactly zero." | **False.** They become negligible relative to the other terms in the same gradient sum — which is why Adam cannot recover them. |
| 2 | "Adam fixes vanishing gradients because it is scale-invariant." | **False.** Scale invariance is *uniform*; vanishing is a ratio between terms inside one parameter's gradient. |
| 3 | "LSTMs solve the vanishing gradient problem." | **False.** They mitigate it via a diagonal, weight-free `C` path; `f` must still drop below 1 to forget, and the `h`-paths are unchanged. |
| 4 | "Gradient clipping fixes exploding and vanishing gradients." | **False.** It caps the upper bound only; it does nothing on the vanishing side. |
| 5 | "A residual connection concatenates its input with the sublayer output." | **False.** It adds: `y = x + F(x)`, giving the `I` term in `∂y/∂x`. Concatenation has no identity path. (The instructor says concatenation — it is wrong.) |
| 6 | "Attention was invented in `Attention Is All You Need`." | **False.** Bahdanau (2014) and Luong (2015) added attention to LSTMs for NMT, two years earlier. |
| 7 | "FlashAttention makes attention `O(n)`." | **False.** It is exact attention with unchanged `O(n²)` FLOPs; it is `O(n)` in *memory* and 2–4× faster in wall clock. |
| 8 | "Multi-head attention adds parameters over single-head attention." | **False.** Both are `4 d_model²`. Multi-head buys diversity of routing, not capacity. |
| 9 | "Positional encodings are needed because the FFN is position-wise." | **False.** They are needed because attention is permutation-equivariant; it treats the input as a set. |
| 10 | "RoPE makes context extension free." | **False.** It encodes relative position, but the model is still trained at a fixed length; extension needs interpolation plus continued pretraining. |
| 11 | "GQA reduces attention FLOPs." | **False.** It reduces KV-cache size and memory bandwidth: Llama-3-8B goes 512 → 128 KB/token, 4×. |
| 12 | "`ln 2 = 0.693` at a flat loss on binary classification means the model is converging." | **False.** It means the model predicts the base rate. The notebook's `0.6942` is exactly this. |
| 13 | "`ln(8000) = 8.987` on a flat loss means the data is fine and the model is broken." | **False.** It means the targets are uniform — the notebook's `np.random.randint` labels. Data bug. |
| 14 | "Teacher forcing eliminates exposure bias." | **False.** It *creates* it: training on ground-truth prefixes, inference on self-generated ones. |
| 15 | "A GRU has 2/3 of an LSTM's parameters because it has 2 of 3 gates." | **False.** It has 3 weight matrices vs 4 → 3/4. At `d_h = 256, d_x = 128`: 296 448 vs 394 240. |
| 16 | "An LSTM has 4× the parameters of a vanilla RNN of the same widths." | **True.** Four gates, each with a `d_h × (d_h + d_x)` matrix. |
| 17 | "Truncated BPTT lets an RNN learn dependencies longer than the truncation window." | **False.** Gradients never cross the segment boundary; only detached state does. |
| 18 | "Exploding gradients are fixed by switching to bf16." | **Partly.** bf16 has fp32's exponent range, so it survives overflow that NaNs fp16 — but the underlying divergence is still there. |
| 19 | "Attention weights are explanations of model behaviour." | **False.** They are one intermediate in a computation with residuals, LayerNorm, multiple heads and an FFN; ablation shows low-weight positions can matter more. |
| 20 | "The FFN holds about two thirds of a transformer layer's parameters." | **True.** `8d² / 12d² = 66.7 %` at `d_ff = 4d_model`, and the same with Llama's `8/3` SwiGLU. |
| 21 | "Post-LN needs a long warmup because LayerNorm slows convergence." | **Partly.** It is because the identity path is inside the LayerNorm, making the effective gradient depth- and variance-dependent. |
| 22 | "Mamba has no KV cache at inference." | **True.** The state is `O(1)` per layer — this is its main serving advantage. |
| 23 | "Mamba's parallel scan gives the state log-depth access to the past." | **False.** The scan is log-depth in *compute*; the information path is still a linear recurrence that overwrites. |
| 24 | "Pure Mamba beats transformers at associative recall at matched scale." | **False.** Recall is the SSM's weakness; that is why every production model is a hybrid. |
| 25 | "An RNN and a transformer can do identical FLOPs and differ 100× in wall clock." | **True.** Arithmetic intensity: 1 FLOP/byte (matvec) vs `n` FLOP/byte (GEMM) vs ~153 needed to saturate an A100. |
| 26 | "ULMFiT was a transformer." | **False.** A 3-layer AWD-LSTM, 2018, no attention — and it beat models trained on 100× more data with 100 labelled examples. |
| 27 | "`padding='post'` is the right default for an LSTM classifier that reads the final state." | **False.** The final state is then a function of `<PAD>`. Use `'pre'` or an explicit `Masking` layer. |
| 28 | "Keras `model.save()` persists the tokenizer." | **False.** It does not. Reloading with a different `num_words` silently remaps token indices — the notebook's exact failure. |
| 29 | "`√d_k` scaling is an empirical trick with no derivation." | **False.** `Var(q·k) = d_k` under unit-variance components, so `sd = √d_k`; the scaling restores unit variance and keeps the softmax Jacobian alive. |
| 30 | "At 512 tokens with `d_model = 4096`, attention dominates the FLOP budget." | **False.** The FFN is `8nd²` vs attention's `4n²d`; they cross at `n ≈ 2d_model ≈ 8192`. |

---

## Coding / Whiteboard Tasks

### Task 1 — Write BPTT for a vanilla RNN from scratch (no autograd)

**Prompt:** "Implement forward and backward for a single-layer `tanh` RNN with `d_x = 128, d_h = 256`, then print the per-step gradient magnitude `‖∂L/∂h_k‖` against `t − k` for `t = 50`."

**Expected solution sketch:**

```python
import numpy as np

def rnn_forward(x, h0, Wxh, Whh, bh):
    """x: (T, d_x). Returns h: (T, d_h) and the pre-activations for the backward pass."""
    T, _ = x.shape
    h = np.zeros((T, Wxh.shape[0]))
    a = np.zeros((T, Wxh.shape[0]))
    h_prev = h0
    for t in range(T):
        a[t] = Wxh @ x[t] + Whh @ h_prev + bh
        h[t] = np.tanh(a[t])
        h_prev = h[t]
    return h, a

def rnn_backward(dh_last, h, a, x, h0, Wxh, Whh):
    """dh_last: (d_h,) gradient of the loss w.r.t. the FINAL hidden state only.
    Returns dWxh, dWhh, dbh and the per-step gradient norms."""
    T, d_h = h.shape
    dWxh = np.zeros_like(Wxh); dWhh = np.zeros_like(Whh); dbh = np.zeros(d_h)
    dh_next = dh_last
    norms = np.zeros(T)
    for t in reversed(range(T)):
        dh = dh_next                                   # gradient arriving from t+1
        norms[t] = np.linalg.norm(dh)                  # ||dL/dh_t||
        da = dh * (1 - h[t] ** 2)                      # tanh'
        dWxh += np.outer(da, x[t])
        dWhh += np.outer(da, h[t - 1] if t > 0 else h0)
        dbh  += da
        dh_next = Whh.T @ da                           # dL/dh_{t-1} = W_hh^T diag(1-h^2) dL/dh_t
    return dWxh, dWhh, dbh, norms

# Reproduce the bound: ||dL/dh_k|| should decay like (gamma*rho)^(t-k)
rng = np.random.default_rng(0)
d_x, d_h, T = 128, 256, 50
Wxh = rng.normal(0, 1 / np.sqrt(d_x), (d_h, d_x))
Whh = rng.normal(0, 1 / np.sqrt(d_h), (d_h, d_h))     # Xavier: spectral radius ~ 1.0
x   = rng.normal(0, 1, (T, d_x))
h, a = rnn_forward(x, np.zeros(d_h), Wxh, Whh, np.zeros(d_h))
_, _, _, norms = rnn_backward(np.ones(d_h), h, a, x, np.zeros(d_h), Wxh, Whh)
for k in [0, 10, 20, 30, 40, 49]:
    print(f"t-k = {49 - k:2d}   ||dL/dh_k|| = {norms[k]:.3e}")
```

**What the interviewer is grading:**
- Does the backward loop carry `dh_next = Whh.T @ da` — i.e. does the candidate know the Jacobian is `diag(1 − h²) W_hhᵀ` and not something else?
- Do they accumulate `dWhh` with `np.outer(da, h_prev)` rather than `da * h_prev`?
- Do they compute `norms[t]` **before** overwriting `dh_next`, so the curve is the gradient at each step and not the running sum?
- The expected output is a *geometric decay* — with Xavier init the printed norms should fall by orders of magnitude between `t−k = 0` and `t−k = 49`. A candidate who is surprised by this has not internalised the bound.

**Common failures:** putting `tanh'` on the wrong side of `W_hhᵀ`; forgetting that `dh_next` must be `dL/dh_{t−1}` not `dL/da_{t−1}`; using `h[t]` instead of `h[t−1]` in the `dWhh` outer product.

---

### Task 2 — Derive the LSTM cell-state gradient and prove it is diagonal

**Prompt:** "Starting from the LSTM equations, compute `∂C_t/∂C_{t−1}` symbolically and say why it has no weight matrix. Then compute `∂h_t/∂h_{t−1}` and explain why *that* is the term that matters for the layers below."

**Expected solution sketch (whiteboard, no code):**

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t

f_t = σ(W_f [h_{t-1}; x_t] + b_f)
i_t = σ(W_i [h_{t-1}; x_t] + b_i)      <- all three depend on h_{t-1}, x_t
g_t = tanh(W_g [h_{t-1}; x_t] + b_g)      NEVER on C_{t-1}

=> dC_t/dC_{t-1} = diag(f_t)             (the i_t ⊙ g_t term contributes 0)

dC_t/dh_{t-1} = diag(C_{t-1}) · df_t/dh_{t-1}
              + diag(g_t)     · di_t/dh_{t-1}
              + diag(i_t)     · dg_t/dh_{t-1}

h_t = o_t ⊙ tanh(C_t)
dh_t/dC_t     = diag(o_t ⊙ (1 - tanh^2(C_t)))
dh_t/dh_{t-1} = diag(tanh(C_t)) · do_t/dh_{t-1}          (sigmoid' <= 0.25)
              + diag(o_t (1-tanh^2 C_t)) · dC_t/dh_{t-1} (contains W_f, W_i, W_g)
```

**What the interviewer is grading:**
- The **zero** on the `i_t ⊙ g_t` term — this is the crux, and a candidate who misses it has not derived it.
- Understanding that `dh_t/dh_{t-1}` contains `W_f, W_i, W_g` and `σ' ≤ 0.25` and is therefore *structurally an RNN Jacobian*. This is the answer to "why doesn't the highway solve it".
- Whether they mention that the same argument applies to `dh_t/dh_{t−1}` reaching *earlier layers* — the gradient to the embedding passes through `h`, not through `C`, so the embedding gradient is governed by the bad Jacobian regardless of the highway.

**Common failures:** claiming `∂C_t/∂C_{t−1} = f_t` without the `diag`; claiming the highway makes `dh_t/dh_{t−1}` diagonal; forgetting `o_t` in the `dh/dC` term.

---

### Task 3 — Implement scaled dot-product attention and verify the `√d_k` claim empirically

**Prompt:** "Write attention in 20 lines of NumPy, then show empirically that the softmax entropy and the gradient magnitude depend on `d_k` if you omit the scaling."

**Expected solution sketch:**

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, scale=True):
    d_k = Q.shape[-1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    A = softmax(scores)
    return A @ V, A

rng = np.random.default_rng(0)
n = 512
for d_k in [16, 64, 128, 512]:
    Q = rng.normal(0, 1, (n, d_k))
    K = rng.normal(0, 1, (n, d_k))
    V = rng.normal(0, 1, (n, d_k))

    raw = Q @ K.T
    print(f"d_k={d_k:4d}  Var(q.k) measured={raw.var():8.1f}  theory={d_k:6d}")

    _, A_unscaled = attention(Q, K, V, scale=False)
    _, A_scaled   = attention(Q, K, V, scale=True)
    # entropy of the attention distribution: low entropy = saturated = dead gradient
    ent = lambda A: -(A * np.log(A + 1e-12)).sum(-1).mean()
    print(f"          entropy unscaled={ent(A_unscaled):.3f}  scaled={ent(A_scaled):.3f}"
          f"   (max = ln {n} = {np.log(n):.3f})")
```

**Expected output:** `Var(q·k)` tracks `d_k` closely; the unscaled entropy collapses toward 0 as `d_k` grows (a near-one-hot distribution with `∂softmax/∂z = p(1−p) ≈ 0`), while the scaled entropy stays near the middle of the achievable range.

**What the interviewer is grading:**
- Correct `max`-subtraction in the softmax (numerical stability).
- Whether they *measure* the variance rather than asserting it.
- Understanding that the deliverable is "the softmax is saturated and its Jacobian is dead", not "the numbers are big".
- Bonus: mentioning that the `max`-subtraction is what makes the online-softmax trick in FlashAttention possible.

**Common failures:** scaling by `d_k` instead of `√d_k`; forgetting that `Q @ K.T` needs `K.T` and not `K`; dividing *after* the softmax.

---

### Task 4 — Size a model: parameters, FLOPs, KV cache

**Prompt:** "Given `n_layers = 32`, `d_model = 4096`, `n_heads = 32`, `n_kv_heads = 8`, `d_ff = 11008`, `vocab = 32000`, compute the total parameters, the per-layer attention and FFN FLOPs at `n = 8192`, and the KV cache at 128 k tokens in fp16. Then say which of the three would force you to change the architecture."

**Expected solution sketch:**

```python
d_model, n_layers, n_heads, n_kv_heads = 4096, 32, 32, 8
d_ff, vocab, d_head = 11008, 32000, d_model // n_heads   # 128

# ---- parameters
attn  = 4 * d_model * d_model                       # W_Q, W_K, W_V, W_O (full-width K,V here)
ffn   = 3 * d_model * d_ff                          # SwiGLU: gate, up, down
norm  = 2 * d_model
per_layer = attn + ffn + norm
embed = vocab * d_model
total = n_layers * per_layer + embed + d_model
print(f"attention/layer {attn:,}  ffn/layer {ffn:,}  layer {per_layer:,}")
print(f"total {total:,}  ({total/1e9:.3f} B)")
print(f"split: ffn {ffn/per_layer:.1%}  attn {attn/per_layer:.1%}  embed {embed/total:.2%}")

# ---- FLOPs at n = 8192 (forward, 2 FLOPs per MAC)
n = 8192
attn_flops = 4 * n * n * d_model + 8 * n * d_model * d_model   # scores+mix + the 4 projections
ffn_flops  = 2 * n * ffn                                       # 3 matrices -> 6*n*d*d_ff, 2 FLOPs/MAC
print(f"attn {attn_flops/1e12:.1f} TFLOP   ffn {ffn_flops*n_layers/1e12:.1f} TFLOP")

# ---- KV cache at 128k tokens, fp16
tokens, bpe = 131072, 2
kv_bytes = 2 * n_layers * n_kv_heads * d_head * bpe * tokens
print(f"KV cache: {kv_bytes/2**30:.1f} GiB   ({kv_bytes/tokens/1024:.0f} KiB/token)")
```

**What the interviewer is grading:**
- The **factor of 2** in the KV cache (keys *and* values). This is the single most common error.
- Knowing that `W_K`/`W_V` shrink with GQA (`n_kv_heads × d_head × d_model`) while `W_Q`/`W_O` do not.
- Recognising the conclusion: the **KV cache** is what forces the architectural change — 32 GB at 128 k for a single sequence, at 128 KiB/token. Attention FLOPs are a compute problem you can buy your way out of with better kernels; the cache is a memory wall at fixed model size.
- Bonus: noting that dropping to `n_kv_heads = 8` *is* the architectural change (GQA), and that fp8 KV quantisation is the other lever.

---

### Task 5 — Diagnose the companion notebook's failure chain

**Prompt:** "Here is a training run: `val_accuracy: 0.5180, val_loss: 0.6926` after `fit(batch_size=64, epochs=1, validation_split=0.1)` on 3 000 rows. Then a second run reports `loss: 8.9872, accuracy: 1.2756e-04`. Explain both numbers and name the bug in each."

**Expected solution sketch (written, not code):**

| Observation | The number | Diagnosis |
|---|---|---|
| `val_loss 0.6926`, `val_acc 0.5180` | `ln 2 = 0.6931` | Balanced binary classifier predicting the base rate. `3000 / 64 = 47` optimizer steps in 1 epoch — underfit, not converged. The "model" is an untrained network plus a coin. |
| `predict(x[0]) = 0.4902` → "Negative" | `≈ 0.5` | The label printed to the user is noise. |
| `loss 8.9872` | `ln 8000 = 8.9871968` | Targets are `np.random.randint(1, 8000, ...)` — uniform noise. Uniform softmax is the Bayes-optimal predictor. |
| `accuracy 1.2756e-04` | `1/8000 = 1.25e-04` | Exactly the accuracy of sampling from the uniform distribution over 8000 classes. |
| `loss 0.6876` on the 1000-row rerun | vs `ln 2 = 0.6931` | A marginal improvement from 15 steps — consistent with fitting the base rate slightly better, not with learning the task. |
| `len(x_train)` prints 5000 after slicing to 3000 | — | Stale cell output; the reproducibility warning. `num_words` changed 10000 → 1000 while the embedding kept 10000 rows and the saved model's embedding was trained on the old index mapping. |

**What the interviewer is grading:**
- The reflex to convert every reported loss into a **baseline comparison**. `ln 2`, `ln V`, `1/V` should be instant.
- Whether they identify the *random targets* as the root cause in the second run rather than proposing training fixes.
- Whether they notice the vocabulary/`num_words` mismatch — the specific mechanism by which the "fine-tune" silently corrupts the frozen embedding.
- Whether they distinguish "the model is broken" from "the data is broken". These require opposite fixes.

---

### Task 6 — Design the long-range test, then interpret the curve

**Prompt:** "You have a model that claims 32 k context. Design the experiment that tells you whether it *uses* it, and describe the three shapes the resulting accuracy-vs-depth curve can take and what each means."

**Expected solution sketch:**

```python
# Position-controlled needle test. The ONLY thing that varies is the needle's depth.
def build_prompt(needle, haystack_tokens, depth_frac, rng):
    """depth_frac=0.0 -> needle at the very start; 1.0 -> at the very end."""
    n = len(haystack_tokens)
    pos = int(depth_frac * (n - 1))
    toks = haystack_tokens[:pos] + needle + haystack_tokens[pos:]
    question = "\n\nWhat is the access code? Answer with the code only."
    return detokenize(toks) + question

rows = []
for depth in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    accs = []
    for trial in range(50):
        needle  = f"The access code is {rng.integers(10000, 99999)}."
        gold    = ...                                  # the code, tracked separately
        prompt  = build_prompt(needle, haystack, depth, rng)
        out     = model.generate(prompt, max_new_tokens=8, do_sample=False)
        accs.append(extract_code(out) == gold)
    rows.append((depth, np.mean(accs), np.std(accs) / np.sqrt(len(accs))))

# CONTROL: same positions, needle replaced by a same-length random string.
# The control's accuracy is the model's format/guessing prior -- subtract it.
```

**The three shapes and what each means:**

| Curve shape | Reading | Failure mode |
|---|---|---|
| **Flat at ceiling** | The model genuinely uses the full context; every depth is equally addressable. | None — but verify the control is at chance, or the task is trivially answerable from local context. |
| **U-shape (high at both ends, sagging in the middle)** | "Lost in the middle". Ends are over-represented in training (or attended via the attention sink / recency bias) and the centre is under-visited. | Optimisation + positional-encoding failure. Fix: length-balanced training data, position interpolation, or retrieval front-loading at serve time. |
| **Monotone decay with distance** | Classic distance failure. Compare the slope against `(γρ)^{Δ}` — a geometric slope says gradient-limited; a steeper-than-geometric slope with a healthy gradient says the **state** cannot hold the content (information failure) or the model has no incentive (optimisation failure). | Distinguish by instrumenting `‖∂L/∂h_k‖` vs `t − k`, which needs gradient access and therefore a training-side experiment. |
| **Chance at every depth** | The model never learned the task, the answer extraction is broken, or the needle tokenization differs by position. | Check the control and read 20 raw generations before concluding anything. |

**What the interviewer is grading:**
- The **control condition**. Without it, an above-chance result is uninterpretable — the model may be exploiting format priors.
- Reporting the **curve**, not the mean. A mean over depths hides the U-shape entirely, and the U-shape is the finding.
- The separation of gradient failure from information failure, which is the whole point of the module reduced to an experimental design.

---

## Cheat Sheet of Numbers To Memorize

**Gradient arithmetic**

| Quantity | Value |
|---|---|
| `0.9^50` | `5.2 × 10⁻³` |
| `0.8^50` | `1.4 × 10⁻⁵` |
| `0.5^50` | `8.9 × 10⁻¹⁶` — below fp32 epsilon |
| `0.99^500` | `0.0066` — an LSTM forget gate at 0.99 loses 99.3 % over 500 steps |
| fp32 machine epsilon | `1.19 × 10⁻⁷` |
| `σ'(·)` maximum | `0.25` (sigmoid); `tanh'` max `1.0` at 0 |
| `σ(1) ≈ 0.731` | the LSTM forget-bias init value `b_f = 1` |
| Xavier std for `d_h = 256` | `1/√256 = 0.0625` → spectral radius `0.0625 × 16 = 1.0` |
| Adam default ε | `1e-8` — the floor below which scale invariance breaks |
| Typical clip threshold | `1.0` (global norm) — RNNs, LSTMs, and Llama-family LLMs alike |

**Architecture and parameters**

| Quantity | Value |
|---|---|
| Attention params per layer | `4 d_model²` |
| FFN params per layer (SwiGLU, `d_ff = 8/3 d_model`) | `8 d_model²` |
| Layer total at `d_model = 4096` | `202 383 360` |
| Llama-2-7B total | `6 738 415 616` (exact published figure) |
| Llama-2-7B split | FFN 64.2 %, attention 31.9 %, embeddings 3.89 % |
| LSTM params, `d_h = 256, d_x = 128` | `394 240` (the notebook's exact number) |
| GRU params, same widths | `296 448` (24.8 % smaller) |
| Vanilla RNN, same widths | `98 560` |
| Notebook model total | `1 674 497` trainable (6.39 MB); `5 023 493` with optimizer state (19.16 MB) |
| Embedding table, notebook | `10 000 × 128 = 1 280 000` params |

**Attention and cost**

| Quantity | Value |
|---|---|
| Attention FLOPs per layer | `≈ 4 n² d_model` |
| FFN FLOPs per layer | `≈ 2 n d_model d_ff` = `8 n d_model²` at `4×` |
| Attention/FFN crossover | `n ≈ 2 d_model` (≈ 8192 at `d_model = 4096`) |
| Score matrix, `n = 20 000`, 32 heads, fp16 | `20000² × 32 × 2 = 25.6 GB` |
| KV cache per token, Llama-3-8B (GQA, 8 KV heads) | `128 KB` → 16 GB at 128 k |
| KV cache per token, Llama-2-7B (32 KV heads) | `512 KB` → 64 GB at 128 k |
| A100-80GB fp16 peak | `312 TFLOP/s`, `2.04 TB/s` HBM |
| Roofline crossover | `≈ 153 FLOP/byte` |
| Arithmetic intensity: matvec / GEMM(n) | `1` / `n` FLOP per byte |
| Measured MFU: RNN / transformer | `~0.6 %` / `~50 %` |
| FlashAttention speedup | 2–4× forward; FLOPs unchanged, memory `O(n²) → O(n)` |
| `√d_k` at `d_k = 128` | `11.3` — the unscaled logit standard deviation |

**Loss baselines — the reflex numbers**

| Baseline | Value |
|---|---|
| Balanced binary cross-entropy | `ln 2 = 0.6931` |
| Uniform over 8 000 classes | `ln 8000 = 8.9872` |
| Uniform over 32 000 classes | `ln 32000 = 10.3735` |
| Uniform over 50 257 classes | `ln 50257 = 10.8249` |
| Uniform sampling accuracy over `V` classes | `1/V` (e.g. `1/8000 = 1.25 × 10⁻⁴`) |
| Perplexity from loss | `exp(loss)` — loss 8.9872 → ppl 8000 |

**Training and data**

| Quantity | Value |
|---|---|
| Notebook classification run | 3 000 rows, batch 64, 1 epoch → **47 optimizer steps**, 89 s |
| Notebook rerun | 1 000 rows, batch 64, 1 epoch → **15 steps**, 22 s |
| Notebook seq2seq run | 1 000 rows, batch 64, 1 epoch → 29 steps, 67 s |
| IMDb truncation | `maxlen = 200`, post-padding, post-truncating |
| Typical warmup for Post-LN | 4 000 steps (original paper) |
| Typical warmup for Pre-LN | 100–2 000 steps |
| ULMFiT headline | 18–24 % error reduction; 100 labelled examples ≈ 100× more data |
| Mamba hybrid ratio (Jamba) | ~1 attention layer per 7 Mamba layers |

---

## Answers To The Self-Check Questions From CS-05

These mirror §19 of CS-05. Answer them aloud in under two minutes each; the full version is what a senior interviewer expects.

**S1. Write the BPTT gradient for `∂L/∂W_hh` in a simple RNN and explain why large `t−k` terms are exponentially small.**
`∂L/∂W_hh = Σ_{t=1}^{T} Σ_{k=1}^{t} [ ∂L_t/∂h_t · Π_{j=k+1}^{t} diag(1 − h_j²) W_hhᵀ ] h_{k−1}ᵀ`. The outer sum runs over output steps, the inner over the source steps influencing each. The bracketed factor is a product of `t−k` Jacobians; by submultiplicativity `‖Π‖ ≤ (γρ)^{t−k}` with `γ = max_j (1 − h_j²) ≤ 1` and `ρ = σ_max(W_hh)`. When `γρ < 1` the term decays geometrically — at `γρ = 0.8` and `t−k = 50` it is `1.4 × 10⁻⁵`, and at `0.5` it is `8.9 × 10⁻¹⁶`, below fp32 epsilon. Crucially the term is still *present* in the sum; it is negligible **relative to** the `k = t` term, which is `O(1)`. So the gradient is dominated by short-range contributions, and the optimizer minimises the empirical risk using them — long-range credit assignment is lost to arithmetic, not to architecture.

**S2. Give the exact per-step gradient factor for the LSTM cell path, derive why it contains no weight matrix, and give two reasons it still does not solve vanishing gradients.**
`∂C_t/∂C_{t−1} = diag(f_t)`. Derivation: `C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t`; both `f_t` and the product `i_t ⊙ g_t` are functions of `h_{t−1}` and `x_t` only — the equations for `f, i, g, o` contain no `C_{t−1}` — so the second term's derivative with respect to `C_{t−1}` is exactly zero, leaving the diagonal `diag(f_t)`. No `W_·` appears because the recurrence is element-wise. Two reasons it does not solve vanishing: (i) `f_t` is *learned* and must be able to fall below 1 to forget anything — the task's need to discard context directly erodes the highway, and `σ` bounds `f ∈ (0,1)` but provides no lower bound away from 0; (ii) the paths that carry gradient to earlier *layers* and to the embeddings run through `h_t = o_t ⊙ tanh(C_t)`, so `∂h_t/∂h_{t−1}` still contains `W_f, W_i, W_g` multiplied by `σ'`/`tanh'` factors `≤ 0.25` — structurally an RNN Jacobian. The output gate also sits on the on-ramp: `∂h_t/∂C_t = o_t ⊙ (1 − tanh²(C_t))`, so a small `o_t` blocks gradient from reaching the highway at all.

**S3. Why `1/√d_k`? Give the variance calculation.**
Let the components of `q, k ∈ ℝ^{d_k}` be i.i.d. with mean 0 and variance 1. Then `E[q_i k_i] = 0` and `Var(q_i k_i) = E[q_i²] E[k_i²] = 1`. The dot product is a sum of `d_k` independent zero-mean terms, so `Var(q·k) = Σ_{i=1}^{d_k} Var(q_i k_i) = d_k`, hence `sd(q·k) = √d_k` — `11.3` at `d_k = 128`. Unscaled, the softmax logits scatter over roughly ±11 standard deviations, so the distribution saturates to near one-hot and its Jacobian `∂softmax/∂z = diag(p) − ppᵀ` collapses (the diagonal entries go to `p(1−p) ≈ 0`), and no gradient reaches the projections. Dividing the logits by `√d_k` restores unit variance. Note the scaling is by `√d_k`, not `d_k`: over-scaling flattens the distribution toward uniform and costs accuracy.

**S4. The instructor says a residual connection concatenates input and sublayer output. What does it actually do, and what breaks if you concatenate?**
It **adds**: `x_{l+1} = x_l + F(LN(x_l))` (Pre-LN) or `LN(x_l + F(x_l))` (Post-LN). Concatenation would (a) grow the width by `d_model` per block — 33× after 32 layers at `d_model = 4096`, i.e. 135 168 — so every downstream parameter count changes and the model is a different model; (b) destroy the gradient property, because `∂(x ⊕ F(x))/∂x` is a block matrix `[I ; ∂F/∂x]` whose product over `L` layers does not retain an identity path — the `I` term in `∂(x + F)/∂x = I + ∂F/∂x` is exactly what guarantees a gradient route of magnitude 1 to the input; and (c) force every subsequent layer to distinguish "original" from "added" content by position in the vector, which is a harder learning problem than addition. Addition is only possible because `F` is width-preserving, which is a constraint on the sublayer design, not an accident.

**S5. 20 000-token context, `O(n²)` attention: compute score-matrix memory in fp16 for 32 heads, then explain what FlashAttention changes.**
`n² × h × bytes = 20000² × 32 × 2 = 8 × 10⁹ × 32 × 2 = 25.6 GB` for a single score matrix, and a training step holds several such intermediates simultaneously (pre-softmax scores, post-softmax probabilities, the gradient of each), so peak is `≈ 75–100 GB` per layer. FlashAttention tiles `Q`, `K`, `V` into blocks small enough for on-chip SRAM (192 KB per SM on an A100), computes the softmax blockwise with the online-softmax running rescale — maintaining running max `m` and running denominator `ℓ`, rescaling previously accumulated output when a new block raises `m` — and never materialises the `n × n` matrix in HBM. Memory becomes `O(n)`; HBM traffic drops to `O(n²d²/M)`; wall clock improves 2–4× forward and more in the backward pass. It changes **nothing** about the FLOPs (`4n²d_model` per layer, still quadratic) and it is **exact** — the only numerical difference is floating-point reassociation, not approximation.

**S6. Give the five independent reasons the notebook's LSTM classifier cannot be repurposed for summarization, each with the line of code that shows it.**
1. **Task type / objective.** Classification ends in `Dense(1, activation='sigmoid')` trained with `binary_crossentropy` (cell 8, cell 11); summarization needs `Dense(8000, activation='softmax')` over a `(batch, 50, 8000)` tensor with `sparse_categorical_crossentropy` (cells 26–27).
2. **Architecture.** `LSTM(latent_dim, return_state=True)` with `return_sequences=False` discards every per-step output `y_t`; a summarizer needs the full output sequence on both encoder and decoder — the notebook has to construct a *new* decoder LSTM with `return_sequences=True` (cell 25).
3. **Vocabulary.** `num_words=vocab_size` with `vocab_size=10000` builds an embedding of 10 000 rows (cells 2–3), but the "retrain" cell reloads the data with `vocab_size=1000` (cell 17) while the frozen embedding still has 10 000 rows — every token ranked 1000–10000 now maps to index 2 (`<UNK>`), so the embedding receives a distribution it never trained on.
4. **OOV / tokenization.** `imdb.load_data` is a word-level index with no subword units and no saved vocabulary file; cell 14 reconstructs the mapping with a hardcoded `+3` offset from `imdb.get_word_index()`. There is nothing you can ship alongside the weights.
5. **Objective mismatch.** BCE on a scalar per example versus sparse categorical CE over a 50-step sequence with an 8000-way softmax per step — different loss surface, different head, different gradient scale (`1/64` vs `1/(64 × 50 × 8000)` averaging).

**S7. Derive the loss of exactly `8.9872` in the notebook's summarization run. What does it tell you about the data?**
The decoder targets are `np.random.randint(1, 8000, (1000, 50, 1))` — independent uniform draws over 8000 classes. A model that learns the marginal distribution assigns probability `1/8000` to each class at every position, giving per-token cross-entropy `−Σ_{c=1}^{8000} (1/8000) ln(1/8000) = ln 8000`. Numerically `ln 8000 = ln 8 + ln 1000 = 2.0794415 + 6.9077553 = 8.9871968`, which rounds to the reported `8.9872`. The accuracy `1.2756 × 10⁻⁴` is `≈ 1/8000 = 1.25 × 10⁻⁴` — exactly the accuracy of sampling from the target distribution. So the model achieved the **Bayes-optimal loss for an unlearnable target**: there is no signal in the labels, and the run is a perfect negative control rather than a failed translation experiment. It is the cleanest accidental demonstration in the course of why the approach failed — and nothing in the notebook flags it.

**S8. Compute the LSTM's arithmetic intensity at one time step and the crossover batch size for an A100 in fp16; explain the ~100× wall-clock gap in one sentence.**
At one time step the LSTM computes `4` gate matvecs, each `2 d_h (d_h + d_x)` FLOPs, so `≈ 8 d_h (d_h + d_x)` FLOPs; it reads the same number of fp16 weight bytes (`2` bytes per weight, i.e. `4 d_h (d_h + d_x)` weights → `8 d_h (d_h + d_x)` bytes). Ratio: **1 FLOP per byte**. An A100-80GB delivers `312 × 10¹²` fp16 FLOP/s against `2.039 × 10¹²` bytes/s, so it needs `312/2.039 ≈ 153` FLOP per byte to be compute-bound — meaning you need a batch/sequence dimension of `n ≳ 153` to saturate it. A transformer folds all `n` positions into a GEMM, so its intensity is `n` FLOP/byte: at `n = 4096` it is 27× past the crossover and runs near peak, while the RNN step stays at 1/153 of peak — under 1 %. In one sentence: **the two architectures do comparable FLOPs per token, but the RNN's FLOPs arrive as a sequence of tiny bandwidth-bound matvecs with a kernel launch between each, so the GPU spends its time moving weights rather than multiplying them.** With ~0.6 % MFU versus ~50 %, the wall-clock gap is about two orders of magnitude — the ~100× figure is the measured consequence, not a rhetorical one.

**S9. Why does RoPE generalize to unseen positions when sinusoidal absolute encoding does not? What property must the encoding have?**
RoPE rotates dimension pairs of `q` and `k` by angles proportional to their absolute position: for a pair `(q_{2i}, q_{2i+1})` it applies a rotation `R_{m,θ_i}` with `θ_i = 10000^{−2i/d}`. Because rotation matrices compose and `(R_m q)ᵀ(R_n k) = qᵀ R_{n−m} k`, the attention score is a function of `m − n` alone. The model therefore learns a *function of the relative offset* — a translation-invariant kernel — and such a function evaluated at a new offset `Δ' > Δ_train` is an interpolation/extrapolation of a function it has already fit, rather than a new input pattern. Sinusoidal **absolute** encoding adds `PE(pos)` to the embedding, so a position beyond `L_train` produces a token representation the model has never been trained on; there is no relative structure in the *input* to generalize from (even though the sinusoidal functions themselves are defined at all positions — the failure is that downstream weights learned to interpret particular patterns). The required property: **the attention score must be a function of relative position only**, i.e. `f(q, m, k, n) = g(q, k, m − n)`. ALiBi achieves this additively (`−m·(i−j)`), RoPE multiplicatively. Both still degrade in practice beyond a modest multiple of `L_train`, which is why extension uses NTK-aware or YaRN frequency interpolation plus continued pretraining.

**S10. Name two tasks where you would deploy an LSTM in 2026 instead of a transformer, with the specific number that decides it.**
1. **Wake-word / keyword spotting on a microcontroller** (e.g. Cortex-M33, 256 KB RAM). A streaming transformer at 200 frames needs a KV cache of `2 × n_layers × n_kv_heads × d_head × 2 B × 200`; at 4 layers, 4 KV heads, `d_head = 64` that is `2 × 4 × 4 × 64 × 2 × 200 = 819 200 B ≈ 800 KB` — more than three times the entire RAM budget, before weights. A 2-layer GRU with `d_h = 48` carries **384 bytes** of state (`2 × 48 × 4 B`) and ~24 KB of INT8 weights, running at ~0.4 ms per frame. The deciding numbers are 800 KB versus 384 bytes, and ~24 KB of weights versus a 7 B-parameter model.
2. **Very-long-context streaming under a hard memory bound**, e.g. log or telemetry analysis at 128 k–1 M tokens per stream. Llama-3-8B's KV cache is 128 KB per token → 16.8 GB at 128 k tokens, more than the 16 GB of fp16 weights themselves, and it grows without bound as the stream continues. An LSTM/SSM's per-stream state is constant — a few hundred kilobytes regardless of stream length — so the same GPU holds orders of magnitude more concurrent streams. The deciding number is that the cache scales with `n` while the recurrent state does not, so at large `n` it is a memory wall rather than a cost difference. The honest caveat: if the task requires exact recall of an arbitrary earlier token, the LSTM will fail where the transformer succeeds, and you should use a hybrid — or a retrieval layer in front of the LSTM.

---

## Cross-References

| For | Go to |
|---|---|
| Full derivations, notebook walkthrough, cost tables | **CS-05** — `case-studies/CS-05-RNN-LSTM-to-Attention.md` |
| Formulas in one-page reference form | **CH-05** — `cheat-sheets/CH-05-RNN-LSTM-Transformers.md` |
| Pretraining, tokenizers, the LLM lifecycle | **CS-01 / IQ-01** |
| Transfer learning & fine-tuning fundamentals | **CS-02** |
| When to fine-tune at all | **CS-04 / IQ-04 / CH-04** |
| BERT fine-tuning (the first post-attention win) | **CS-07 / IQ-07 / CH-07** |
| LoRA/QLoRA (where the adapters go, and why attention projections) | **CS-13 §6.8** / **CS-11 §4.11** (CS-23 planned, not yet written) |
| Continued pretraining for long context | **CS-12** |
| Quantisation of weights and the KV cache | **CS-10 / CS-11** |

