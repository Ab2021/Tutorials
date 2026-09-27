# CS-05 — Why Fine-Tuning Was Hard Pre-Transformer: RNN/LSTM → Attention

| Field | Value |
|---|---|
| **Module** | Foundations / Architecture History |
| **Source video(s)** | "LLM Fine-Tuning 06: Why Finetuning Was Difficult in RNN or LSTM – How Transformers Changed the Game"; "LLM Fine-Tuning 07: LSTM vs Transformer \| Why Transformers Replaced LSTM in NLP" |
| **Transcript file(s)** | `LLM_Fine-Tuning_06_Why_Finetuning_Was_Difficult_in_RNN_or_LSTM_How_Transformers.txt`, `LLM_Fine-Tuning_07_LSTM_vs_Transformer_Why_Transformers_Replaced_LSTM_in_NLP.txt` |
| **Companion code** | `LLM Fine-Tuning-05-Why-Finetuning-Hard-in-LSTM\Why_finetuning_was_challanging_in_LSTM (1).ipynb` |
| **Prerequisites** | CS-01 (pretraining & lifecycle), CS-02 (transfer learning), CS-04 (FT vs RAG vs agents) |
| **Difficulty** | Intermediate (the derivations are the hard part; the intuition is not) |
| **Hands-on required** | Yes — the notebook is 30 lines and reproduces in 3 minutes on a free Colab T4 |
| **Estimated study time** | 4h theory + 1.5h practical |

---

## 0. Executive Summary

- **Two independent problems killed the RNN era, and they are usually conflated.** (1) *Gradient*: backprop-through-time (BPTT) multiplies the same Jacobian `n` times, so the gradient to step `k` scales like `λ^(t−k)`. At λ = 0.9 and a 50-step gap the factor is `5.2 × 10⁻³`; at λ = 0.5 it is `8.9 × 10⁻¹⁶`, i.e. numerically dead in fp32. (2) *Throughput*: `h_t` depends on `h_{t−1}`, so you cannot batch across time. A matvec has arithmetic intensity ≈ 1 FLOP/byte; an A100 needs ≈ 150 FLOP/byte to saturate its tensor cores. **The RNN runs at ~0.6 % of peak; the transformer at 40–55 %.** Fixing (1) alone (LSTM gates) changes nothing about (2).
- **The single most important number in this module:** for a fixed per-step contraction λ, the gradient arriving from `n` steps away is `λⁿ`. 50 steps is the cliff. `0.9⁵⁰ = 5.2e-3` (survivable, barely), `0.8⁵⁰ = 1.4e-5` (dominated by Adam's ε), `0.5⁵⁰ = 8.9e-16` (dead).
- **The LSTM cell state is a gradient highway, not a solution.** `∂C_t/∂C_{t−1} = diag(f_t)` contains **no weight matrix and no saturating nonlinearity**, so it can carry a gradient for thousands of steps *in principle*. In practice `f_t` must drop below 1 to forget, so `Π f_j` still decays. And the hidden-state path `∂h_t/∂h_{t−1}` — the one that actually feeds the next layer and the output — still contains `U_f, U_i, U_o, U_g` and still saturates through `tanh`.
- **The instructor's "RNN remembers 10–15 words, LSTM 30 words" [06:15:35–15:54] is a folk heuristic.** The measurable numbers are: Bengio et al. (1994) showed a simple RNN loses the gradient beyond ~10–20 steps; Khandelwal et al. (2018) measured LSTM LMs using ~200 tokens of effective context with a sharp ~50-token window; and with orthogonal init + LayerNorm + gradient clipping, LSTMs train on 1000+ step sequences. See §10.1.
- **The encoder–decoder bottleneck is an information problem, not a gradient problem.** A 50-token source at `d = 512` is 25 600 floats compressed into a 1024-float `(h, C)` pair — a 25× compression that no amount of gating fixes. Bahdanau attention (2014) removed it; on WMT'14 En→Fr, BLEU for the no-attention model collapses as sentence length grows past 30 while the attention model stays flat.
- **Why fine-tuning specifically was impossible:** every item on the list in §8 has to hold for transfer learning to work, and in 2014–2017 *none* of them did — no pretrained checkpoint, no shared tokenizer, no shared architecture, no framework, datasets 4–7 orders of magnitude too small, and a per-task retrain cost paid from scratch every time.
- **The instructor's own notebook proves the point better than his narration does, in three places:** the classifier reaches **50.11 % accuracy** on binary sentiment after one epoch (a coin flip); the retrain loads the same weights under a **10× smaller vocabulary**, which silently collapses most tokens to `<UNK>`; and the "summarization" run trains on `np.random.randint` targets and lands on a loss of **8.9872 = ln(8000) exactly** — the uniform-distribution loss, i.e. the model output nothing and could not have. §6.5 derives that number.
- **ULMFiT (2018) is the proof that the *idea* of pretrain→fine-tune worked on LSTMs.** It got 18–24 % error reduction on most datasets and matched from-scratch training on 100× more data using only 100 labeled examples. It failed to become universal for one reason: **it was still sequential**, so it could not be scaled to the corpora that make pretraining pay.
- **Do not over-generalize the other way.** Recurrent and state-space models still win where they always won: O(1) per-token inference state, constant inference memory, streaming audio, MCU-class edge deployment, and RL world models. The 2024–2026 frontier is *hybrid* (Jamba, Zamba, Bamba, Samba, Hymba, Qwen3-Next), not pure-attention.

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world

You are asked to fine-tune a model for a domain classification task. In 2026 you write three lines:

```python
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B", num_labels=4)
trainer = Trainer(model=model, args=TrainingArguments(...), train_dataset=ds)
trainer.train()
```

That works because 15 trillion tokens of general language are already compressed into the weights, the tokenizer already covers your domain's vocabulary at ~99 %+, and the architecture is the same one every other model uses so every tool, kernel, and quantizer supports it.

**None of those three facts were true for an LSTM in 2016.** You wrote the model definition yourself, you ran `Tokenizer.fit_on_texts()` on your own 25 000 rows, you trained the embedding matrix from scratch, and you trained the recurrent weights from scratch, for every task, every time. This module is the explanation of *why*, at the level of the gradient and the FLOP.

### 1.2 What the state of the art was before

| Era | Dominant sequence architecture | Representative result |
|---|---|---|
| 1990–1997 | Simple RNN, BPTT | Bengio et al. 1994: gradient vanishes exponentially with the gap |
| 1997–2014 | LSTM (Hochreiter & Schmidhuber), then GRU (Cho et al. 2014) | LSTM beats RNN on long-lag tasks; still sequential |
| 2014 | Encoder–decoder (Sutskever et al.; Cho et al.) | WMT'14 En→Fr BLEU 34.8 (single model, 5-model ensemble 36.5) |
| 2014/2015 | Encoder–decoder **+ additive attention** (Bahdanau et al.) | BLEU 26.75 on the *harder* WMT'14 En→Fr setup used in that paper; no degradation with length |
| 2017 | Transformer (Vaswani et al.) | 28.4 EN-DE / 41.8 EN-FR BLEU; trained in **12 hours on 8 P100s** for the big model |
| 2018 | BERT, GPT, ULMFiT | Transfer learning becomes the default |
| 2023–2026 | Hybrid attention/SSM, GQA, FlashAttention, 128K–1M context | See §13 and §16 |

The instructor covers this arc explicitly: encoder–decoder in **2014** [06:24:00, 07:22:43], transformer in **2017 by Google** [07:24:10], and "later on the GPT/BERT model which was a transformer-based model replaced LSTM as universal NLP architecture" [06:47:59–48:12].

### 1.3 The naive approach, and precisely why it fails

Naive approach: *treat a sequence as a bag of embeddings and run an MLP.* It fails immediately because word order carries the signal — the instructor's own example pair makes the point, `I am eating apple` vs `... while choosing apple`, where the first *apple* is a fruit and the second is a phone [07:26:01–26:50]. Position 4 vs position 9 changes the referent. So you need order. The obvious way to get order is a recurrence:

```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
y_t = W_hy h_t + b_y
```

This is correct, expressive, and trainable in principle. It fails in practice for exactly two reasons, and this module is the derivation of both.

### 1.4 Concrete motivating example with numbers

The instructor's notebook trains an LSTM sentiment classifier on 3 000 IMDb reviews, one epoch, batch 64 — **89 seconds on a Colab GPU** — and gets:

```
71/71 ━━━ 89s 1s/step - accuracy: 0.5011 - loss: 0.6942 - val_accuracy: 0.5180 - val_loss: 0.6926
```

`0.5011` accuracy and `0.6942` loss. Binary cross-entropy at a constant 50/50 prediction is `−ln(0.5) = 0.6931`. **The model has learned nothing.** He then predicts on a real review, gets `0.4902`, and prints "Negative 😞". The prediction is a coin flip rendered as a sentiment.

That is not a notebook bug — the code is correct and the shapes are correct. It is the *phenomenon*: a 1.67 M-parameter LSTM, trained from scratch on 3 000 examples, with a 200-step BPTT path, cannot learn in one epoch. Every design decision in the transformer — parallelism, attention, residuals, LayerNorm — is a response to the forces that produced this table.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: a relay race where each runner is a lossy photocopier

Imagine `n` runners in a line. Runner 1 carries a message. He hands a *photocopy* to runner 2, who photocopies that and hands it to runner 3, and so on. Two things happen:

1. **The message degrades multiplicatively.** If each copier has fidelity 0.9, after 50 handoffs the message is at 0.9⁵⁰ = 0.5 % strength. If fidelity is 0.8, it is 0.0014 %. The *loss is not additive, it is multiplicative* — that is the entire vanishing-gradient story.
2. **The race is serial.** Runner 51 cannot start until runner 50 has finished. You have 1000 runners and one track. Adding more runners does not make the race finish sooner. That is the sequential-computation story.

The LSTM's cell state is a **separate sealed tube** running alongside the runners that only gets *added to* rather than replaced — the message in the tube survives because nobody re-copies it. That is `C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t`: when `f_t ≈ 1`, the tube passes the payload through untouched.

The transformer's answer is to **abolish the relay entirely**. Every token gets a direct phone line to every other token, and the "message" is a weighted average of what everyone says. The path length between any two tokens is 1. The cost is that everyone must be on the call at once — `n²` lines.

### 2.2 Where these analogies break

- **The photocopier analogy implies the gradient is uniformly attenuated.** It is not: the Jacobian is a *matrix*, and the relevant quantity is its largest singular value, not a scalar fidelity. An RNN whose `W_hh` has `σ_max > 1` in one direction and `< 1` in another explodes along one axis and vanishes along another *simultaneously*. Real RNNs do both at once, which is why you need clipping (for the exploding axis) *and* gating (for the vanishing axis).
- **The "sealed tube" analogy implies the cell state is lossless.** It is not: `f_t` is a *learned sigmoid* and the network has every incentive to push it below 1 in order to forget. Measured forget-gate activations in trained LSTMs sit around 0.5–0.9 for most units, so `Π f_j` still decays — more slowly than `Π diag(tanh') W_hh`, but it decays.
- **The "phone line" analogy implies attention is free.** It is `O(n²)` in both FLOPs and (pre-FlashAttention) memory. And it implies the transformer has solved long-range dependency, which is only partly true: attention gives **path length 1** but not **capacity** — you still need a head to have learned to route, and the residual stream is a finite-width bus that all `n` tokens write into.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Sequence data** | Data where the order of elements changes the meaning: text, time series, audio, DNA, sensor streams [06:07:31–07:43; 07:04:20–05:41] | Defines the problem class the whole module addresses | "Any tabular data with a time column" is *not* necessarily sequence data if the order is not semantically load-bearing |
| **RNN** (recurrent neural network) | A neural network whose hidden layer's output is fed back as an additional input to the same hidden layer at the next time step [06:06:31–07:27] | The first architecture that could consume variable-length sequences | "Recurrent" does **not** mean "deep" — `N` stacked RNN layers is a different axis from `T` time steps |
| **Recurrency / the loop** | The feedback edge `h_{t−1} → h_t`; the instructor calls it "the loop" [07:07:08–07:29] | This single edge is the source of *both* problems: it is what BPTT unrolls (gradient) and what forbids parallelism (throughput) | People think the loop is a memory feature. It is, but it is also the bottleneck |
| **Unrolled RNN** | The same weights drawn once per time step; `T` copies sharing one parameter set [06:08:00–09:38] | Makes the shared-Jacobian argument visible: BPTT differentiates through `T` uses of the *same* `W_hh` | The unrolled graph looks like a deep feedforward net, but it is *weight-tied* — that is the crucial difference |
| **Hidden state `h_t`** | The `d_h`-dimensional vector carrying all information from steps `< t` | The entire memory of the model; a fixed-size bottleneck over a variable-length past | Confusing `h_t` (the state) with `y_t` (the output). In many-to-one classification only `h_T` is used |
| **Many-to-one** | Sequence in, single output out. Text classification / sentiment [06:18:00–19:16] | The notebook's task | "Many" refers to time steps, not to a batch dimension |
| **Many-to-many (synchronized)** | Sequence in, same-length sequence out. NER, POS tagging [06:19:43–20:38] | Requires `return_sequences=True` | Not the same as seq2seq |
| **Many-to-many (asynchronized)** | Sequence in, different-length sequence out. Summarization, translation, generation [06:21:21–22:26] | Needs encoder–decoder; this is the regime where the bottleneck bites hardest | People call every many-to-many "seq2seq". It is specifically the *asynchronous* one |
| **Encoder–decoder** | Two networks: an encoder consumes the source and emits a final state; a decoder is *initialized* from that state and generates the target [06:22:43–24:40; 07:22:43] | Sutskever et al. / Cho et al., 2014 | The confusion: the encoder's *output* is discarded; only `(h_T, C_T)` crosses the boundary. That is the bottleneck |
| **BPTT** (backpropagation through time) | Reverse-mode autodiff on the unrolled graph | Produces the `Π ∂h_j/∂h_{j−1}` product that vanishes/explodes | "BPTT" is not a different algorithm from backprop; it is backprop on a specific graph |
| **Truncated BPTT (TBPTT)** | Backprop only `k` steps, then stop the gradient (`detach`) | Makes training feasible; makes the gradient estimate biased and caps learnable dependency at `k` | People think TBPTT "solves" vanishing gradients. It does not — it *hides* them by never asking for a long-range gradient |
| **Vanishing gradient** | `‖∂h_t/∂h_k‖ → 0` exponentially in `t−k` | The reason simple RNNs cannot learn long-range dependencies [06:16:30; 07:10:21] | Not the same as "the loss is small". A model can have a tiny loss and a dead gradient |
| **Exploding gradient** | `‖∂h_t/∂h_k‖ → ∞` exponentially; NaNs, loss spikes | The other half of the same mechanism; fixed by clipping, not by gating | People fix vanishing gradients with clipping, which does nothing for vanishing |
| **Gradient clipping** | Rescale `g` so `‖g‖ ≤ c` (by global norm) before the optimizer step | The standard exploding fix; also lets you push the learning rate | **Clip-by-value** (`clipvalue`) breaks the gradient *direction*; always prefer clip-by-global-norm |
| **Forget gate `f_t`** | `σ(...)`; multiplies the previous cell state element-wise [06:13:11–13:35; 07:14:02] | The gradient highway's only valve | The instructor says the forget gate does "cross multiplication" [07:17:50] — it is an **element-wise (Hadamard)** product, not a cross product. See §10.4 |
| **Input gate `i_t`** | `σ(...)`; gates the candidate write into the cell state [06:13:22] | Controls what new information enters long-term memory | The input gate and the candidate `g_t` are *separate*; the write is `i_t ⊙ g_t`, not `i_t` |
| **Output gate `o_t`** | `σ(...)`; gates how much of `tanh(C_t)` is exposed as `h_t` [06:13:39–13:45] | Decouples "what I remember" from "what I emit" | People think `h_t = C_t`. It is `o_t ⊙ tanh(C_t)` — the memory is *not* directly readable |
| **Cell state `C_t`** | The additive long-term memory channel; "long-term memory highway" [07:15:18–15:23] | The gradient highway: `∂C_t/∂C_{t−1} = diag(f_t)`, no weight matrix | It is not a solution to vanishing gradients in general — see §4.3.2 |
| **GRU** (gated recurrent unit) | Two gates (update `z_t`, reset `r_t`); merges `h` and `C` into one state [Cho et al. 2014] | 25 % fewer recurrent params than LSTM; comparable quality on most tasks | Not covered in the video at all. See §4.3.3 |
| **Sequence-to-sequence (seq2seq)** | Any mapping from a sequence to a sequence [06:17:07–23:07] | The umbrella term | "Seq2seq" and "encoder–decoder" are *not* synonyms: many-to-many-synchronized is seq2seq without an encoder–decoder |
| **Teacher forcing** | Feeding the *ground-truth* previous token as the decoder input during training [06:39:08–40:00; 07:34:47–35:57] | Makes decoder training parallel and stable | The instructor calls it "teacher forcing" correctly, but the notebook **does not implement it** — it feeds random integers. §6.5 |
| **Exposure bias** | The train/inference mismatch created by teacher forcing | At inference the model sees its own errors, which it never saw in training | The standard mitigation is scheduled sampling, not "more epochs" |
| **Attention** | A learned, content-based weighted average over a set of vectors | Gives path length 1 between any two positions; the core of the transformer | Attention is not "explainability" — attention weights are not faithful explanations |
| **Self-attention** | Attention where Q, K, V all come from the same sequence [06:26:33; 07:29:33] | The transformer's replacement for recurrence | The video calls it "self potential" in places — transcription artifact for "self-attention" |
| **Q, K, V** | Query, Key, Value: three *learned linear projections* of the input embedding [07:40:53–43:12] | `W_Q, W_K, W_V` are trainable; `Q/K/V` are activations | Q/K/V are not "three copies of the input". They are projections through three different matrices |
| **Scaled dot-product attention** | `softmax(QKᵀ/√d_k)V` [07:41:53–43:12] | The `1/√d_k` is the only non-obvious part | The scaling is not for numerical range of the output; it is to keep the *softmax input* variance at 1. §4.5.2 |
| **Masked (causal) attention** | Attention where position `i` cannot see `j > i`; implemented by adding `−∞` to future scores [07:36:25–37:53] | Required for autoregressive training to avoid "cheating" | The mask is applied to the *scores before softmax*, not to the output |
| **Cross-attention** | Attention where Q comes from the decoder and K, V come from the encoder output [07:38:02–39:02] | The only place encoder and decoder interact in the transformer | Distinct from self-attention in *provenance* of K and V, not in formula |
| **Multi-head attention (MHA)** | `h` parallel attention heads on `d_k = d_model/h` subspaces, outputs concatenated and projected | Lets different heads attend to different relations | Concatenation + `W_O` is not the same as averaging h independent attentions |
| **Residual connection** | `x + Sublayer(x)` [07:31:00–31:49] | Preserves a gradient path of length 1 through the block | The instructor says input and sublayer output are "concatenated" [07:31:15]. It is element-wise **addition**. Concatenation would grow `d_model` every block. See §10.2 |
| **LayerNorm** | Normalizes across the feature dimension per token, then applies learned `γ, β` | Stabilizes training; makes `d_model` and depth decoupled | LayerNorm normalizes over *features within one token*, BatchNorm over *tokens within one feature* |
| **Pre-LN vs Post-LN** | `x + F(LN(x))` vs `LN(x + F(x))` | Pre-LN removes the warmup requirement; Post-LN is the original paper's layout [07:31:51–32:20] | The video shows only Post-LN. Every modern LLM is Pre-LN or uses RMSNorm pre-norm |
| **FFN** | Position-wise 2-layer MLP with a 4× expansion; SwiGLU in modern models [07:31:51–32:17] | ~2/3 of all transformer parameters. §4.6 | "Feed forward" here means per-token, not across tokens |
| **`Nx`** | The repeat count for identical blocks; 6 in the original paper [07:32:22–32:48] | Depth is the main quality lever | The blocks are *not* identical in parameters — identical in *structure* |
| **Positional encoding** | Information added to embeddings so a parallel model knows order [07:28:39–29:14] | Necessary precisely because the transformer dropped recurrence | Sinusoidal → learned → RoPE → ALiBi. §4.5.4 |
| **RoPE** | Rotary position embedding: rotate Q and K by position-dependent angles | Relative-position property; the default in Llama/Mistral/Qwen | RoPE is applied to Q and K only, never to V |
| **ALiBi** | Additive linear bias `−m·(i−j)` to scores; no positional embedding at all | Better length extrapolation; used in BLOOM/MPT | ALiBi is not "attention without position" — the bias *is* the position |
| **KV cache** | Stored K and V from all previous positions, to avoid recompute at decode | Makes autoregressive inference `O(n)` per token instead of `O(n²)` | The cache is a *memory* cost, not a compute saving — it is what makes inference memory-bound |
| **SSM** (state-space model) | `x_k = Āx_{k−1} + B̄u_k`, `y_k = Cx_k`; a linear recurrence | Trainable in parallel as a convolution; `O(1)` inference state | Not an RNN in the gated sense, and not a transformer |
| **Mamba / S6** | Selective SSM: `B, C, Δ` become input-dependent | Breaks the convolution trick but enables a hardware-aware parallel scan | Mamba is **not** "an upgraded transformer" (the instructor says it is, [06:11:49]). It is a different family. §10.5 |
| **FlashAttention** | IO-aware exact attention; tiles into SRAM, never materializes `n×n` | Cuts attention memory from `O(n²)` to `O(n)`; 2–4× wall-clock | It does **not** change the FLOP count. Still `O(n²)`. §16.2 |
| **ULMFiT** | Universal Language Model Fine-tuning; 2018; AWD-LSTM | Proved pretrain→fine-tune works on LSTMs; the direct ancestor of the transformer-era recipe | The instructor says it was "introduced along with the transformer" [06:46:55]. It is 2018, one year *after*. §10.6 |
| **Bottleneck** | Forcing a variable-length input through a fixed-size vector | The encoder–decoder `(h_T, C_T)` handoff | Not the same as the *architectural* bottleneck in a residual stream; here it is information-theoretic |
| **Catastrophic forgetting** | Fine-tuning on task B destroys performance on task A [07:47:08–47:28] | The reason "train one model on everything" needs either replay, regularization, or capacity | The instructor is describing *sequential* fine-tuning of the same weights without any of the modern mitigations (LoRA, replay, EWC) |

---

## 4. Deep Dive — How It Actually Works

### 4.1 Notation

| Symbol | Meaning | Typical value in this module |
|---|---|---|
| `T` / `n` | sequence length (time steps) | 200 (notebook `max_len`), 50 (worked example) |
| `d_x` | input/embedding dimension | 128 (`embedding_dim`) |
| `d_h` | hidden / recurrent state dimension | 256 (`latent_dim`) |
| `d_model` | transformer width | 4096 (Llama-2-7B) |
| `h` | number of attention heads | 32 (Llama-2-7B) |
| `d_k = d_model / h` | per-head key/query dimension | 128 |
| `x_t ∈ ℝ^{d_x}` | input at step `t` | |
| `h_t ∈ ℝ^{d_h}` | hidden state at step `t` | |
| `C_t ∈ ℝ^{d_h}` | LSTM cell state | |
| `⊙` | Hadamard (element-wise) product | |
| `σ` | logistic sigmoid `1/(1+e^{−z})`, `σ' = σ(1−σ) ∈ (0, 0.25]` | |
| `λ` | per-step gradient scale factor, `‖∂h_j/∂h_{j−1}‖` | 0.5–1.5 in practice |

### 4.2 Mechanism, step by step — the simple RNN

**Forward.** At each step `t`:

```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)      h_0 = 0
y_t = W_hy h_t + b_y
```

`W_xh ∈ ℝ^{d_h×d_x}`, `W_hh ∈ ℝ^{d_h×d_h}`, `b_h ∈ ℝ^{d_h}`. **The same three tensors are used at every step.** That weight tying is what makes the model a *recurrent* network and it is the whole source of the trouble.

Parameter count for the recurrence:
`|W_xh| + |W_hh| + |b_h| = d_h·d_x + d_h² + d_h`.
With `d_h = 256, d_x = 128`: `32768 + 65536 + 256 = 98 560`.

**Backward (BPTT).** Let `L = Σ_{t=1}^{T} L_t` be the total loss. Define the per-step Jacobian

```
J_j  ≡  ∂h_j/∂h_{j-1}  =  diag(1 − h_j²) · W_hhᵀ            ∈ ℝ^{d_h×d_h}
```

(the `diag(1 − h_j²)` term is `tanh'` evaluated at the pre-activation; `1 − h_j² ∈ (0, 1]`).

Then for the gradient of the loss at step `t` with respect to the hidden state at step `k < t`:

```
∂L_t/∂h_k  =  ∂L_t/∂h_t · Π_{j=k+1}^{t} J_j
           =  ∂L_t/∂h_t · Π_{j=k+1}^{t} [ diag(1 − h_j²) W_hhᵀ ]
```

and the gradient with respect to the recurrent weight matrix (unrolling and applying the product rule over all `t` uses) is

```
∂L/∂W_hh  =  Σ_{t=1}^{T} Σ_{k=1}^{t}  [ ∂L_t/∂h_t · Π_{j=k+1}^{t} J_j ] · h_{k-1}ᵀ
```

**Read the double sum carefully.** Every term in it contains the product of `(t − k)` Jacobians. The terms where `k` is close to `t` have a short product and contribute normally. The terms where `k` is far from `t` — exactly the terms that would teach the model a long-range dependency — have a long product and are exponentially small. **The long-range gradient is not absent; it is present and drowned.** That distinction matters, because it explains why Adam does not rescue you (§4.2.3).

#### 4.2.1 The eigenvalue / product argument

Take a scalar norm bound. Using submultiplicativity:

```
‖Π_{j=k+1}^{t} J_j‖  ≤  Π_{j=k+1}^{t} ‖J_j‖
                     ≤  ( max_j ‖diag(1 − h_j²)‖ · ‖W_hh‖ )^{t−k}
```

`max_j ‖diag(1 − h_j²)‖ = max_j max_i (1 − h_{j,i}²)`. Since `tanh` output is bounded by 1, `1 − h² ≤ 1`, and the bound is attained only when the unit is exactly at 0. **Away from zero, tanh saturates and the factor shrinks.** Empirically, trained RNNs have a large fraction of hidden units with `|h| > 0.9`, giving per-unit factors of `1 − 0.81 = 0.19`.

So define `γ = max_j ‖diag(1 − h_j²)‖₂ ≤ 1` and `ρ = σ_max(W_hh)` (the largest singular value, i.e. the spectral norm). Then

```
‖∂h_t/∂h_k‖  ≲  (γ · ρ)^{t−k}  =  λ^{t−k}
```

- If `λ < 1` → **vanishing**. Exponentially small in the gap.
- If `λ > 1` → **exploding**. Exponentially large in the gap.
- `λ = 1` exactly is a measure-zero knife edge, and it does not stay at 1 for all directions during training. This is why the same network can show both symptoms in different layers.

**Initialization note.** Xavier/Glorot init sets `Var(W) = 2/(fan_in + fan_out)`. For a `d_h × d_h` matrix that is `2/(2d_h) = 1/d_h`, so per-entry std `σ = 1/√d_h = 1/16 = 0.0625` at `d_h=256`. For a square random matrix with iid entries of std `σ`, the spectral radius concentrates near `σ√d_h = 0.0625 × 16 = 1.0`. **Xavier init deliberately puts an RNN at the vanishing/exploding boundary**, which is the worst possible place to be: the gradient neither decays cleanly nor explodes cleanly, it does a random walk in log-space. The standard fix is orthogonal init (`ρ = 1` by construction) or an explicit spectral-radius rescale, and even those only pin the *norm*; the direction still rotates.

#### 4.2.2 The arithmetic for a 50-step sequence

Take `λ` as the per-step scale. The gradient from a 50-step gap is `λ⁵⁰`:

| `λ` per step | `λ⁵⁰` | Reading |
|---|---|---|
| 0.50 | `8.9 × 10⁻¹⁶` | Dead. Below fp32 epsilon (1.2e-7) by 9 orders of magnitude |
| 0.70 | `1.8 × 10⁻⁸` | Dead. fp32 can represent it, but it is below Adam's `ε = 1e-8` floor |
| 0.80 | `1.4 × 10⁻⁵` | Effectively dead: 5 orders below the short-range terms it competes with |
| 0.90 | `5.2 × 10⁻³` | Attenuated 190×. Survivable only if there is no competing short-range signal |
| 0.95 | `7.7 × 10⁻²` | Attenuated 13×. This is the regime a well-tuned LSTM lives in |
| 1.00 | `1.0` | Knife edge. Passes the gradient through unchanged — and passes the *noise* too |
| 1.05 | `1.15 × 10¹` | Growing. Loss spikes become likely |
| 1.10 | `1.2 × 10²` | Divergence within a few hundred steps |
| 1.20 | `9.1 × 10³` | NaN within tens of steps |
| 1.50 | `6.4 × 10⁸` | Immediate NaN |

**The asymmetry is the punchline.** To keep the gradient alive you need `λ ≥ 0.9`; to keep it stable you need `λ ≤ 1.0`. That is a 10 % window on a quantity you do not directly control and cannot measure cheaply. This is why the RNN era was a hyperparameter-tuning era.

#### 4.2.3 Why Adam does not rescue a vanishing gradient

Adam's update is `θ ← θ − η · m̂/(√v̂ + ε)`. Because of the division by the running RMS `√v̂`, Adam is **scale-invariant**: multiplying the entire gradient by `10⁻¹⁵` produces almost the same update as multiplying it by `1`. A lot of engineers conclude from this that vanishing gradients are a solved problem under Adam. They are not, and the reason is precise:

Look again at `∂L/∂W_hh = Σ_t Σ_k (long-range term) + (short-range term)`. For a *single scalar parameter* of `W_hh`, the total gradient is a sum whose long-range contributions are `~10⁻¹⁵` relative to its short-range contributions. Adam normalizes the **sum**, not the components. The sum is short-range-dominated, so the parameter moves on short-range signal. The long-range component is inside the normalized direction at a relative weight of `10⁻¹⁵` — it is gone before normalization, not after.

Two corollaries worth memorizing:

1. **Adam cannot manufacture information that has been multiplied away.** Scale invariance recovers a *uniformly* small gradient; it cannot recover a gradient that is small *relative to another term in the same sum*.
2. **`ε = 1e-8` sets a hard floor.** For a parameter whose RMS gradient is below `ε`, Adam's update degenerates to `η · m̂/ε`, i.e. pure sign-following on noise. This is why the `λ = 0.7` row above (1.8e-8, just above ε) is worse than the `λ = 0.5` row: at 1.8e-8 the update is large *and* wrong.

> **Beyond the video:** the modern mitigations, in the order they matter for a *residual* network, are (a) residual connections, which give an *additive* identity path so `∂L/∂h_k` has a term that is exactly `1` regardless of depth; (b) LayerNorm/RMSNorm placed *inside* the residual branch, which keeps activation scales from drifting; (c) careful init that makes `Var(h_{t}) ≈ Var(h_{t−1})` at step 0 (e.g. `W_hh` orthogonal, or `W_xh` scaled by `1/√T` for a sum-pooled output); (d) *not* using tanh as the only nonlinearity. An RNN has access to (c) only. A transformer has all four. That is the structural difference, and it is why the transformer trains with no warmup in the Pre-LN configuration while an RNN needs a learning-rate schedule tuned per dataset.

### 4.3 The LSTM: gates, the highway, and why it is not a fix

#### 4.3.1 The full gate equations

Let `x_t ∈ ℝ^{d_x}` be the input, `h_{t−1} ∈ ℝ^{d_h}` the previous hidden state, `C_{t−1} ∈ ℝ^{d_h}` the previous cell state. Concatenate `[h_{t−1}; x_t] ∈ ℝ^{d_h+d_x}` and write the four affine maps:

```
Forget gate      f_t = σ( W_f x_t + U_f h_{t-1} + b_f )        (what to erase)
Input gate       i_t = σ( W_i x_t + U_i h_{t-1} + b_i )        (how much to write)
Candidate        g_t = tanh( W_g x_t + U_g h_{t-1} + b_g )     (what to write)   [a.k.a. c̃_t]
Output gate      o_t = σ( W_o x_t + U_o h_{t-1} + b_o )        (how much to expose)

Cell state       C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t               (additive update — the highway)
Hidden state     h_t = o_t ⊙ tanh( C_t )                       (the gated read-out)
Output           y_t = W_hy h_t + b_y
```

with `W_· ∈ ℝ^{d_h×d_x}`, `U_· ∈ ℝ^{d_h×d_h}`, `b_· ∈ ℝ^{d_h}`.

**The instructor's mapping to these equations** is exact and worth quoting because he gets the *semantics* right even though he skips the algebra:

- forget gate: "the purpose of the forget gate is what needs to be forgot" [06:13:28–13:33]
- input gate: "the main purpose of input gate what needs to be add… what new thing is going to be add" [06:13:33–13:37]
- output gate: "what needs to be passed as a output to the next time step" [06:13:39–13:45]
- cell state: "cell state means long-term memory highway… it's going to be stored the long-term memory" [07:15:18–15:23]
- "in the forget [gate] we are doing a cross the multiplication… and here in the input [gate] we are doing addition" [07:17:53–18:03] — the *multiplication for forget, addition for write* structure is correct and is the key insight.

**Parameter count:**

```
|LSTM| = 4 × ( d_h·d_x  +  d_h²  +  d_h )          (4 gate-equivalents, 1 bias each)
```
With `d_h = 256, d_x = 128`: `4 × (32768 + 65536 + 256) = 4 × 98560 = 394 240`.

This matches the notebook's `classification_model.summary()` exactly: `lstm_3 (LSTM) … 394,240`. **Use that number as your sanity check whenever you re-derive an LSTM size.** If PyTorch reports `4·d_h·(d_x + d_h + 1)` instead, it is because `nn.LSTM` uses *two* bias vectors (`b_ih` and `b_hh`) with the second initialized to zero:
`4 × (d_h·d_x + d_h² + 2·d_h) = 4 × (32768 + 65536 + 512) = 395 264`.
Both conventions exist; Keras fuses to one bias, PyTorch keeps two.

#### 4.3.2 The gradient highway, derived

Differentiate the cell-state recurrence with respect to `C_{t−1}`:

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t

∂C_t/∂C_{t-1} = diag(f_t)  +  [ ∂f_t/∂C_{t-1} terms ]  +  [ ∂i_t/∂C_{t-1}, ∂g_t/∂C_{t-1} terms ]

             = diag(f_t)          ← because f_t, i_t, g_t depend on h_{t-1}, not on C_{t-1}
```

This is the entire trick, and it is worth stating in one sentence: **`f_t`, `i_t`, `g_t` are functions of `h_{t−1}` and `x_t` only, and `h_{t−1} = o_{t−1} ⊙ tanh(C_{t−1})` breaks the direct dependence — but critically, the *dominant* path `∂C_t/∂C_{t−1} = diag(f_t)` contains no weight matrix and no `tanh'`.**

Compare the two products for a gap of `t − k` steps:

```
Simple RNN:   ∂h_t/∂h_k  =  Π  diag(1 − h_j²) · W_hhᵀ              ← d_h²-matrix per factor
LSTM (cell):  ∂C_t/∂C_k  =  Π  diag(f_j)                          ← diagonal, no weights
```

So the LSTM cell path is:
- a **diagonal** product, not a matrix product — no `U_f, U_i, U_o, U_g` appear;
- with entries in `(0, 1)` because `f_j = σ(·)`, so it **cannot explode**;
- and it passes `≈ 1` when the network learns `f_j ≈ 1`.

**Now the four reasons it is nevertheless not a solution.**

1. **`f_t` is learned and the network wants it below 1.** To forget, a unit must lower `f_t`. Empirically, mean forget-gate activations in trained LSTMs range roughly 0.5–0.9 across units, so `Π f_j` over 50 steps is `0.9⁵⁰ = 5.2e-3` at best and `0.5⁵⁰ = 8.9e-16` at typical. **The highway has exactly the same exponential structure as the RNN — it just has a better base.** The gain over a plain RNN is real (a factor of `(0.9/(γρ))⁵⁰`), but it is a constant-factor gain in the exponent, not a change of regime.

2. **The gradient that matters most does not flow through `C`.** The gradient to the *input embeddings* and to *earlier layers* flows through `h`, not `C`. Differentiate the read-out:

```
h_t = o_t ⊙ tanh(C_t)
∂h_t/∂h_{t-1} = diag(o_t ⊙ (1 − tanh²(C_t))) · ∂C_t/∂h_{t-1}
              + diag(tanh(C_t)) · ∂o_t/∂h_{t-1}
```
and `∂C_t/∂h_{t-1} = diag(C_{t-1})·∂f_t/∂h_{t-1} + diag(g_t)·∂i_t/∂h_{t-1} + diag(i_t)·∂g_t/∂h_{t-1}`.
Every one of those `∂f/∂h, ∂i/∂h, ∂o/∂h, ∂g/∂h` contains `U_·` and a `σ'` or `tanh'` factor of at most `0.25`. **So `∂h_t/∂h_{t−1}` is a sum of `d_h`-wide matrix products with saturating factors — structurally the same as the RNN's `J_j`.** The cell-state highway only helps the *cell-to-cell* gradient, which by itself is useless: nothing reads `C` except `h`.

3. **`tanh(C_t)` saturates again on the way out.** `C_t` can grow large over a long sequence — that is the *point* of an additive memory. But once `|C_t| ≳ 3`, `tanh'(C_t) < 0.01`, and the gradient through the read-out dies. This is why `C_t` must be *bounded* to be readable, which reintroduces a forgetting pressure.

4. **Clipping is still mandatory.** Because the forget-gate path is diagonal and bounded, the *cell* path cannot explode — but the gate Jacobians in reason (2) can, and the output layer can. Every production RNN training script from 2015 to 2020 had `clip_grad_norm_(model.parameters(), 0.25)` or `1.0` in it. PyTorch's default for `nn.utils.clip_grad_norm_` is `max_norm=1e9` — i.e. **clipping is off by default and you have to turn it on.** A large fraction of "my LSTM won't train" reports trace to exactly this.

> **Correction to the framing:** the LSTM does not "solve" the vanishing gradient. It replaces `Π diag(tanh') W_hhᵀ` with `Π diag(f)` on one path and leaves the other paths unchanged. The correct statement — the one that survives an interview — is: **gating converts an unbounded, matrix-valued contraction into a bounded, diagonal one, which makes the *decay rate* a learnable, per-unit quantity rather than a function of the weight matrix's spectrum. It does not remove the exponential.** Empirically this buys roughly one order of magnitude in usable dependency length (10–20 steps → 100–200 steps), which is why it was the state of the art for 17 years and why it is not the state of the art now.

#### 4.3.3 GRU — same trick, fewer parts

```
Update gate   z_t = σ( W_z x_t + U_z h_{t-1} + b_z )
Reset gate    r_t = σ( W_r x_t + U_r h_{t-1} + b_r )
Candidate     n_t = tanh( W_n x_t + U_n (r_t ⊙ h_{t-1}) + b_n )
Hidden        h_t = (1 − z_t) ⊙ n_t  +  z_t ⊙ h_{t-1}
```

The hidden-state update is *directly additive*: when `z_t → 1`, `h_t = h_{t-1}` exactly, and

```
∂h_t/∂h_{t-1} ⊇ diag(z_t)
```

So the GRU puts an identity-like path **on the vector that everything actually reads** — which is arguably a cleaner design than the LSTM's separate `C` channel that must pass through `tanh` and a gate before it is readable.

| | LSTM | GRU |
|---|---|---|
| Gate-equivalents | 4 (`f, i, o, g`) | 3 (`z, r, n`) |
| States | 2 (`h`, `C`) | 1 (`h`) |
| Params at `d_h=256, d_x=128` | `4 × 98560 = 394 240` | `3 × 98816 = 296 448` |
| Ratio | 1.00 | **0.752** |
| Highway path | `∂C_t/∂C_{t−1} = diag(f_t)`, then `tanh` + `o_t` on read-out | `∂h_t/∂h_{t-1} ⊇ diag(z_t)`, directly readable |
| Typical quality | Reference | Within ±0.5 BLEU / ±0.3 % accuracy on most tasks; slightly better on small data |
| Inference state per layer | `2·d_h` floats | `d_h` floats |

**Decision rule:** use GRU when you are parameter- or latency-constrained at small scale; use LSTM when you need the separate memory channel (long-form sequence labeling, character-level modelling, music). For fine-tuning *an LLM* neither is relevant — this table exists because interviewers ask it and because it is the correct prior for edge/streaming work (§13, §16.4).

#### 4.3.4 The three ways long-range dependencies break (they are different)

This is a favourite interview discriminator because candidates collapse three distinct failures into one.

**(a) Gradient failure (distance-in-steps).** The BPTT product of §4.2.1. The gradient signal decays as `λ^{t−k}`. Fixed only by architecture (residuals, attention) or by shortening the distance.

**(b) Information failure (fixed-size state bottleneck).** Even with a *perfect* gradient — imagine `λ = 1` exactly, no decay at all — you still have to fit the entire source into `h_T ∈ ℝ^{d_h}` (or `(h_T, C_T) ∈ ℝ^{2d_h}`). Information-theoretically, a `d`-dimensional real vector can distinguish at most `~d` independent directions. A 50-token source at `d_x = 512` carries 25 600 floats of embedding content into 512 floats of state (using `d_h = 512`): a **50× compression** that must be lossless for the decoder to reconstruct arbitrarily. It cannot be. This is why the encoder–decoder's BLEU *collapses with length* rather than degrading gracefully — the failure is not about distance, it is about **total content exceeded capacity**.

The instructor demonstrates his own version of (b) in code: the encoder passes exactly `state_h, state_c` to the decoder and nothing else — `decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])` [notebook cell 26]. That is the bottleneck, in one line.

**(c) Optimization failure (no incentive to use long context).** If the training corpus rarely requires a 200-step dependency, gradient descent will not build the machinery for it — the short-range solution already drives the loss down. Khandelwal et al. (2018) showed this directly: LSTM LMs *have* long-range machinery available but use it weakly beyond ~50 tokens because natural text rarely rewards it. **This is why "my LSTM handles 200 words" is an architecture claim and "my LSTM uses 200 words" is an empirical claim, and they differ.**

> **Beyond the video — the empirical evidence the instructor cites versus the evidence that exists:**
>
> He states RNN memory as "we cannot remember like more than 10 to 15 words" and LSTM as "we can recall that sentence up to 30 words" [06:15:35–15:54, repeated 07:20:13–20:18]. There is no experiment behind those specific numbers. The real results:
>
> | Claim | Measured evidence |
> |---|---|
> | Simple RNN gradient dies beyond ~10–20 steps | Bengio, Simard & Frasconi (1994), *Learning long-term dependencies with gradient descent is difficult* — the original exponential-decay proof |
> | LSTM effective context ≈ 200 tokens, sharp within ~50 | Khandelwal et al. (2018), *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context* — a gate-based probe on a 2-layer LSTM LM, showing a sharp window of ~50 tokens and usable context to ~200 |
> | LSTM can be trained on 1000+ step dependencies | Hochreiter & Schmidhuber's original 1997 constant-error-carousel tasks (1000 steps); modern char-LMs on Penn Treebank and the `pytorch/examples/word_language_model` benchmarks routinely use `bptt=35` but *carry state* between chunks, giving effective dependencies of many hundreds |
> | LSTM's real limit is throughput, not memory | AWD-LSTM (Merity et al. 2017) reached 57.3 perplexity on PTB — competitive with much larger models — on a single GPU, but took ~24 h per run and could not scale to the 40 GB corpora that became standard |
>
> **Interview-safe phrasing:** "A simple RNN's gradient decays as roughly `λ^n` and is unusable past 10–20 steps. An LSTM's cell path decays as `Π f_j` and, with a well-tuned forget gate, carries usable signal to ~100–200 steps — measured, not theoretical. With full BPTT on 1000 steps you can train longer dependencies, but you pay the sequential cost, which is the real reason nobody does."

### 4.4 Why sequential computation is the second, equally fatal problem

The instructor states it in one line: "because of the sequential processing… there is a lack of scalability" [06:47:48–47:50] and "in transformer actually we have parallel processing" [06:47:52]. Here is the arithmetic that makes it fatal.

#### 4.4.1 Arithmetic intensity — the actual mechanism

GPU performance is bounded by two numbers: peak FLOPs and memory bandwidth. The ratio that decides which one binds is **arithmetic intensity** (FLOPs per byte moved):

```
AI = FLOPs / Bytes
A100-80GB-SXM:  312e12 FLOPS (fp16 dense, tensor cores)  /  2.039e12 B/s (HBM2e)
             =  153 FLOP/byte at peak
             ≈  75-100 FLOP/byte at a realistic 50-65% MFU
```

Now compute AI for the two shapes:

- **RNN at one time step = a matrix–vector product.** Weight matrix `4·d_h × d_h` in fp16 = `8·d_h²` bytes; FLOPs = `2 · (4 d_h) · d_h = 8 d_h²`.
  `AI_matvec = 8 d_h² / 8 d_h² = **1 FLOP/byte**`.
- **Transformer processing `n` tokens at once = a matrix–matrix product.** Same weight matrix read once; FLOPs = `2 · (4 d_h) · d_h · n = 8 d_h² n`.
  `AI_gemm(n) = 8 d_h² n / 8 d_h² = **n FLOP/byte**`.

**You need `n ≳ 100` to saturate an A100 in fp16. The RNN structurally has `n = 1`.**

#### 4.4.2 Worked comparison at `d_h = 4096`, one layer, one 4096-token sequence

Weights per gate: `4096 × 4096 = 16.78 M` values. Four gates: `67.1 M` values. In fp16 that is **134 MB** of weights that must be streamed from HBM for every single token.

**RNN path (must be serial):**
- Traffic per token: 134 MB (weights) + ~16 KB (activations, negligible)
- Time per token: `134e6 B / 1.55e12 B/s ≈ 86 µs` (using a realistic 1.55 TB/s achieved bandwidth, not the 2.04 TB/s spec)
- Useful math per token: `8 d_h² = 8 × 16.78e6 = 1.34e8` FLOPs
- Theoretical math time: `1.34e8 / 312e12 = 0.43 µs`
- **GPU utilization: `0.43 / 86 = 0.5 %`**
- Wall clock for 4096 tokens: `4096 × 86 µs ≈ 0.35 s` for **one layer, one sequence**
- Throughput: **~11 600 tokens/s/GPU**

**Transformer path (batched across time):**
- The same 134 MB of weights is read **once** for the whole sequence.
- Math: `8 d_h² n = 1.34e8 × 4096 = 5.5e11` FLOPs
- Time at 312 TFLOPS: `1.76 ms`; at 50 % MFU: `3.5 ms`
- **GPU utilization: 50 %**
- Wall clock for 4096 tokens: `~3.5 ms`
- Throughput: **~1.17 M tokens/s/GPU**

**Ratio: ~100× in wall clock for identical parameter count and identical per-token FLOPs.**

Read that again, because it is the whole argument. The transformer is not doing *less work*. Per token it does `2×` or `3×` the FLOPs of the LSTM (attention + FFN + projections vs. four gates). It finishes **~100×** sooner because it converts a stream of matvecs into one GEMM, and a GEMM at `n = 4096` has `AI = 4096` — 27× above the 153 FLOP/byte crossover, so it is firmly in the compute-bound regime where the tensor cores actually fire.

> **Correction to the FLOPs framing you will hear in interviews:** "RNNs are cheaper than transformers" is *true per token* and *false per second*. An LSTM has ~`8 d²` FLOPs/token/layer against a transformer's ~`24 d²` at short `n`, so the RNN is ~3× cheaper in *arithmetic*. It is ~100× more expensive in *wall clock* because it cannot reach the tensor cores. Never quote FLOPs without stating the utilization; a FLOP you cannot issue is not a FLOP you saved.

#### 4.4.3 Why TBPTT does not fix this

Truncated BPTT (backprop `k` steps, then `detach`) is what everyone actually ran, with `k ≈ 35–200`. It is *necessary* — full BPTT over 4096 steps requires storing 4096 activations per layer and produces the `λ^4096` product. But it has two consequences the video does not mention:

1. **It caps learnable dependency at exactly `k`.** No gradient can cross the truncation boundary, so a dependency of `k+1` steps is invisible. "We use TBPTT with k=35" is a *model capacity statement*, not just an efficiency setting.
2. **It biases the gradient estimate.** The gradient you compute is the gradient of a *different*, shorter-horizon objective. For a stationary signal this is a small bias; for anything with structure longer than `k`, it is a systematic error that no learning-rate tuning repairs.

With stateful chunking (`stateful=True` in Keras, or carrying `h` across batches in PyTorch), you get *forward* context longer than `k` at the cost of a *backward* path still capped at `k`. That mismatch is why inference quality on very long inputs is often better than the training loss suggests — and why it is a trap to evaluate on inputs longer than your training window.

### 4.5 Attention: the replacement

#### 4.5.1 The flow, as the instructor presents it

He walks the encoder in order [07:27:47–28:04] and the summary is accurate:

```
tokenize → token IDs → word embedding → + positional encoding
        → [ multi-head self-attention → residual add → LayerNorm
            → feed-forward → residual add → LayerNorm ] × N
```

and the decoder [07:33:57–39:45]:

```
target tokens (shifted right, <S> prepended) → embedding → + positional encoding
   → masked multi-head self-attention → add & norm
   → cross-attention (Q = decoder, K,V = encoder output) → add & norm
   → feed-forward → add & norm
   → linear (→ vocab_size) → softmax
```

He labels the three attention variants correctly: **self-attention** in the encoder, **masked self-attention** in the decoder, **cross-attention** between them [07:38:23–39:02]. He also gives the correct *reason* for the mask — "so this particular block will be able to see the entire sentence… it should not be like this… step-by-step model should learn… so this is going to be a cheating for the model. So what we do guys? So we hide the future word." [07:36:40–37:30]. That is the correct and complete explanation of causal masking.

His positional-encoding reason is also correct: "in the RNN analysis we were passing the input [as] the sequence so that's why we were able to preserve the sequence of the data but here we are going to be pass the data in parallel — that's why we are not able to preserve the sequence … and we'll have to put this positional encoding over there" [07:28:39–29:14].

He demonstrates self-attention with the canonical paper figure: *"The animal didn't cross the street because it was too tired"* and the attention weight of `it` on `animal` [07:30:25–30:51]. That is Figure 3 of Vaswani et al. (2017), reproduced correctly.

#### 4.5.2 The QKV formulation and the `1/√d_k`

For an input `X ∈ ℝ^{n×d_model}`:

```
Q = X W_Q,   K = X W_K,   V = X W_V            W_Q, W_K, W_V ∈ ℝ^{d_model × d_k}
Scores      = Q Kᵀ  / √d_k                     ∈ ℝ^{n×n}
A           = softmax(Scores, dim=-1)          row-stochastic
Attention   = A V                              ∈ ℝ^{n×d_v}
```

Per head. With `h` heads, each head uses `d_k = d_v = d_model / h`, and:

```
MHA(X) = Concat(head_1, …, head_h) W_O          W_O ∈ ℝ^{d_model × d_model}
```

The instructor's narration of this is correct in structure — "this vector we are going to be multiply with the query [matrix]… we are going to be generate a key vector… multiply with the value [matrix]" [07:40:53–42:42] — and he correctly identifies `W_Q, W_K, W_V` as "random weights… trainable parameter" [07:42:42–42:49] and correctly states the pipeline as "multiply this query and key… we are getting one score. We are going to be normalize it… multiply it with a value vector. And finally we are getting the attention vector" [07:43:00–43:08]. He also gives the reason: "we are doing it for sustaining the global content" [07:43:15–43:19].

He does **not** explain the `1/√d_k`, though he does mention "normalize it using the… divide of this under root k means this is a dimension" [07:41:07–42:14]. The derivation:

Assume the components of `q` and `k` are iid with mean 0 and variance 1 (they are layer-normed or at least roughly standardized). Then

```
q · k = Σ_{i=1}^{d_k} q_i k_i
E[q · k] = Σ E[q_i] E[k_i] = 0
Var(q · k) = Σ Var(q_i k_i) = Σ E[q_i²]E[k_i²] = d_k
std(q · k) = √d_k
```

So the raw logits have standard deviation `√d_k` — at `d_k = 128` that is `11.3`. Softmax with logit std 11.3 is **nearly one-hot**: `softmax([11.3, 0, 0]) ≈ [0.99998, 0.00001, 0.00001]`. Four consequences:

1. **The gradient through softmax vanishes.** `∂softmax/∂z` scales with `p_i(1 − p_i)`; at `p = 0.99998` that is `2 × 10⁻⁵`. Attention stops learning.
2. **The distribution is decided by noise.** Logits that differ by 1 are `0.09σ` apart — invisible.
3. **Entropy collapses**, so the head cannot average over multiple relevant positions.
4. **`d_k` becomes an implicit temperature**, coupling head width to effective sharpness and making `d_model` and `h` non-independent hyperparameters.

Dividing by `√d_k` resets `Var(Scores) = 1`, restoring a softmax operating in its informative range and giving `∂softmax/∂z` a healthy magnitude. **This is the single most-asked "why" question about attention and the answer is a variance argument, not a numerics argument.** The `1/√d_k` is not there to prevent overflow (fp16 overflows at 65504, and `√d_k` scaling does not fix that — the mask's `−∞` and the `−max` subtraction inside the softmax kernel do). It is there to keep the *statistics* of the pre-softmax logits at unit scale.

#### 4.5.3 Multi-head: what it buys and what it does not

The instructor's coverage of MHA is thin — he says "multi" and moves on [07:28:00, 07:38:28]. The precise statement:

- **Total compute is unchanged** versus single-head attention of width `d_model`: `h` heads of width `d_model/h` cost the same `4 n² d_model` FLOPs.
- **Total parameters are unchanged** (`4 d_model²` either way).
- **What changes is the *rank structure* of the attention pattern.** A single head produces one `n × n` row-stochastic matrix, i.e. one softmax-normalized mixture. `h` heads produce `h` independently-normalized mixtures, each in its own `d_k`-dimensional subspace, and the outputs are concatenated and mixed by `W_O`. **Each head's mixing weights independently sum to 1.** That is the real difference: an average is constrained to the convex hull of its inputs; a concatenation of `h` separate averages, linearly recombined, is not.
- **Empirically**, heads specialize. Voita et al. (2019) found a small number of heads in each layer doing identifiable jobs — positional, syntactic, and "rare token" heads — with the *majority* prunable with <4 % BLEU loss on WMT. Olsson et al. (2022) found "induction heads" forming in a phase change early in training, responsible for in-context copying. **Caveat:** this is *descriptive*, not a design principle. You cannot in general predict which heads matter before training, and structured pruning of heads is a research-grade operation, not a routine optimization.

#### 4.5.4 Positional encodings: sinusoidal → learned → RoPE → ALiBi

The original paper's sinusoids [07:28:39–29:14]:

```
PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )
X_final = X_embedding + PE
```

The frequency schedule is geometric from `1` down to `1/10000`; the choice makes `PE(pos+k)` a fixed linear function of `PE(pos)` for any fixed `k` (a rotation), which is *why* the paper hoped it would extrapolate.

| Scheme | Formula / mechanism | Where used | Extrapolation past trained length | Notes |
|---|---|---|---|---|
| **Sinusoidal (absolute)** | `sin/cos` at geometric frequencies, added to embeddings | Original transformer (2017), early BERT variants | Poor; degrades sharply past `L_train` | The paper's own ablation (Table 3, row E) found **learned** position embeddings performed essentially identically: 27.1 vs 27.3 EN-DE BLEU |
| **Learned absolute** | `nn.Embedding(max_len, d_model)`, added | BERT, GPT-2 | Hard cap at `max_len`; no extrapolation | Costs `max_len × d_model` params (1024 × 768 = 786 K for BERT-base — small) |
| **RoPE** (Su et al. 2021) | Rotate Q and K in `d_k/2` 2-D subspaces by angle `pos · θ_i`, `θ_i = 10000^{−2i/d_k}` | Llama 1/2/3, Mistral, Qwen, DeepSeek, PaLM, GPT-NeoX | Extrapolates moderately; extendable via **NTK-scaling / YaRN / ABF** with a short fine-tune | The inner product `⟨RoPE(q,m), RoPE(k,n)⟩` depends **only on `m−n`** — a relative-position property, which is why it generalizes |
| **ALiBi** (Press et al. 2021) | No positional embedding; add `−m·(i−j)` to the score for `j ≤ i`, `m` a fixed per-head geometric slope (`m_h = 2^{−8h/H}`) | BLOOM, MPT, Falcon | Strong; the standard demonstration is train at 1K, evaluate at 2K+ with no fine-tune | The bias *is* the position information; "no positional encoding" is misleading |
| **NoPE** | Nothing | Some 2024–2026 models | Emergent from the causal mask alone | Works because causal attention already breaks permutation symmetry |

> **Beyond the video — what you actually do in 2026:** you do not choose. Llama-family checkpoints ship with RoPE and a `rope_scaling` config (`{"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}`). To extend context you either (a) load a checkpoint already trained at the longer length, (b) apply YaRN/NTK scaling and continue-pretrain on a few billion tokens, or (c) use ALiBi-based models if extrapolation without training is the hard requirement. Changing the positional scheme on a pretrained checkpoint without continued training destroys the model — the Q/K geometry the attention heads learned is a function of the encoding.

#### 4.5.5 Why attention is `O(n²)`, and what it costs

**FLOPs.** Per layer, per head, the two attention matmuls are:

```
QKᵀ : 2 n² d_k  FLOPs      softmax(QKᵀ)V : 2 n² d_v  FLOPs
```
Summing over `h` heads with `d_k = d_v = d_model/h`:
```
Attention FLOPs/layer = 2 n² d_model (QKᵀ) + 2 n² d_model (AV) = 4 n² d_model
```
Compare the linear parts (projections `4 n d_model²`, FFN `2 n d_model d_ff ≈ 8 n d_model²` for `d_ff = 4 d_model`):
```
Linear FLOPs/layer ≈ 12 n d_model²
```
**Attention overtakes everything else when `4 n² d_model > 12 n d_model²`, i.e. when `n > 3 d_model`.** For `d_model = 4096` that is `n > 12 288`. Below 12K tokens, **attention is not the dominant cost — the FFN is.** This is a fact that gets lost in "attention is O(n²)" discourse, and it is why the 128K-context era needed FlashAttention and GQA while the 4K era did not.

**Memory (pre-FlashAttention).** The score matrix is `n × n` per head:
```
n = 8192,  h = 32,  fp16:   8192² × 32 × 2 B = 4.29 GB      ← one intermediate, one layer
n = 32768, h = 32,  fp16:   32768² × 32 × 2 B = 68.7 GB
n = 131072, h = 32, fp16:   131072² × 32 × 2 B = 1.10 TB
```
Four such tensors are live simultaneously in the naive implementation (scores, softmax output, dropout mask, and the `AV` output), so multiply by ~3–4 in practice. **This is what made long context impossible before 2022**, and it is a *memory-bandwidth* problem more than a capacity problem: even if you could store it, writing and re-reading 4.29 GB per layer per forward pass costs `2 × 4.29 GB / 1.55 TB/s = 5.5 ms` of pure traffic, versus 1.76 ms of useful math. **Attention was memory-bound, not compute-bound, which is the opposite of what "O(n²) FLOPs" suggests.** FlashAttention (§16.2) is precisely the fix for that mismatch.

**KV cache (the inference-time `O(n)` memory).** At decode time you cache K and V for all previous positions. Per token, per layer:

```
bytes = 2 (K and V) × n_kv_heads × d_head × bytes_per_element
```

Worked numbers in the table in §4.6.3.

### 4.6 The transformer block end to end

#### 4.6.1 Post-LN (the original) vs Pre-LN (everything modern)

```
Post-LN (Vaswani et al. 2017)          Pre-LN (GPT-2 onward, Llama, Mistral, Qwen)
─────────────────────────────          ────────────────────────────────────────────
z = LN(x + MHA(x))                     z = x + MHA(LN(x))
y = LN(z + FFN(z))                     y = z + FFN(LN(z))
```
The instructor describes the Post-LN layout — "whatever thing we are going to be processed through the multi attention, both thing we are going to be concatenating and then we are passing it to the normalization" [07:31:11–31:28] — and gives the correct *reasons* for the residual: "it help to regulate the gradient flow… stabilize training and speed up the training… to preserve the more context in the training" [07:31:32–31:49].

Why Pre-LN won, mechanically: in Post-LN, the residual stream is renormalized at every block, so the *identity path* `∂x_{l+1}/∂x_l` contains `∂LN/∂x` at every layer — the gradient is rescaled by the norm's Jacobian `l` times. In Pre-LN, the identity path is exactly `I` at every layer, so the gradient reaches layer 1 at full magnitude. Consequence: **Post-LN requires learning-rate warmup (the original paper used 4000 warmup steps) or it diverges; Pre-LN trains without warmup.** Xiong et al. (2020) proved the gradient-norm scaling argument for this.

**RMSNorm** (Zhang & Sennrich 2019) is now standard in place of LayerNorm — it drops the mean-centering and the bias:

```
RMSNorm(x) = x / sqrt( mean(x²) + ε )  ⊙ γ          vs   LayerNorm(x) = (x − μ)/σ ⊙ γ + β
```
~10 % faster and empirically equal; note that **Pre-LN + RMSNorm is a different normalization from Post-LN + LayerNorm**, and you cannot mix a checkpoint's weights with the wrong layout.

#### 4.6.2 Parameter distribution — the worked arithmetic

Take **Llama-2-7B** exactly: `d_model = 4096`, `n_layers = 32`, `h = 32` (`d_head = 128`), `d_ff = 11008` (SwiGLU), `vocab = 32000`, untied embeddings, RMSNorm, bias-free.

Per layer:

```
Attention (MHA, no biases):
    W_Q, W_K, W_V, W_O  =  4 × d_model²      = 4 × 4096²      =  67,108,864
FFN (SwiGLU: gate, up, down):
    3 × d_model × d_ff                       = 3 × 4096 × 11008 = 135,266,304
RMSNorm (2 per layer, weight only):
    2 × d_model                              = 2 × 4096        =       8,192
                                                  per-layer total = 202,383,360
```

Whole model:

```
32 layers × 202,383,360                                     =  6,476,267,520
2 embedding tables (untied: input + output head) 2 × 32000 × 4096 =    262,144,000
final RMSNorm                                                 =          4,096
                                                              ───────────────────
TOTAL                                                          =  6,738,415,616
```

**Llama-2-7B's published parameter count is `6,738,415,616`.** The arithmetic lands on it exactly. Use this whenever you need to size a model from a config.

Now the *distribution*, which is the number that matters for training and for LoRA:

| Component | Parameters | Share | Why it matters |
|---|---|---|---|
| FFN (all layers) | 4,328,521,728 | **64.2 %** | The FFN is where the "knowledge" is stored. LoRA on `q_proj`+`v_proj` alone touches ~2 % of this |
| Attention (all layers) | 2,147,483,648 | **31.9 %** | `W_Q, W_K, W_V, W_O`; LoRA's default target |
| Embeddings (2 tables) | 262,144,000 | **3.9 %** | Rarely fine-tuned; often frozen. Untied doubles this — GPT-2-style tied embeddings save half |
| Norms | 266,240 | **0.004 %** | Never worth touching |

**The 2/3-FFN, 1/3-attention split is stable across essentially the whole modern LLM family.** Two immediate consequences:

1. **LoRA on attention-only adapts a third of the model.** The standard `target_modules=["q_proj","v_proj"]` recipe touches `2 × 4096² = 33.5 M` of 6.74 B parameters — **0.50 %**. It works, but the FFN is where domain *knowledge* lives. `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` covers 96 % of parameters with rank-8 adapters and is the modern default.
2. **A `d_ff = 4·d_model` ReLU/GELU FFN (the original transformer) is `2 d_model d_ff = 8 d_model²` per layer — smaller than SwiGLU's `3 d_model × (8/3) d_model = 8 d_model²`? No: SwiGLU's `d_ff` is chosen as `(8/3)·d_model` precisely so that its 3-matrix FFN costs the same 8 `d_model²` as a 4× 2-matrix FFN.** That is why `11008 ≈ (8/3) × 4096 = 10922.7` rounded up to a multiple of 256. If you see `d_ff = 14336` and `3 × d_model × d_ff`, you are looking at a model that knowingly spent 30 % more on FFN.

#### 4.6.3 Memory accounting at inference — the KV cache

```
KV bytes per token per layer = 2 × n_kv_heads × d_head × bytes_per_element
```

| Model | `n_layers` | `n_kv_heads` | `d_head` | Bytes/token/layer (fp16) | Bytes/token (all layers) |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 128 | `2×32×128×2 = 16 KB` | **512 KB** |
| Llama-2-13B (MHA) | 40 | 40 | 128 | `2×40×128×2 = 20 KB` | **800 KB** |
| Llama-3-8B (GQA 4:1) | 32 | 8 | 128 | `2×8×128×2 = 4 KB` | **128 KB** |
| Llama-3-70B (GQA 8:1) | 80 | 8 | 128 | `2×8×128×2 = 4 KB` | **320 KB** |
| Mistral-7B (GQA 4:1) | 32 | 8 | 128 | 4 KB | 128 KB |

Concrete consequence — **Llama-3-8B at 128K context, batch 1, fp16:**
```
131,072 tokens × 128 KB/token = 16.8 GB
```
That is more than the weights (16.1 GB in fp16) and it fits an 80 GB A100 only at batch ≤ 2 alongside activations. **This is why every long-context deployment in 2026 uses GQA + FP8/INT8 KV cache:** at FP8 the cache halves to 8.4 GB and you can serve 4 concurrent 128K sessions on one 80 GB card.

> **Beyond the video — the KV cache arithmetic the video never does, and the RNN comparison it implies.** The LSTM's inference state is `2 × d_h` floats per layer, *independent of sequence length*. For a hypothetical 32-layer LSTM with `d_h = 4096`: `2 × 4096 × 2 B × 32 = 512 KB total, constant`. Compare Llama-3-8B's 128 KB/**token**. At `n = 8192` the transformer cache is 1.0 GB; the RNN's is 512 KB — a **2000× difference**, and it does not grow. That is the genuine, permanent advantage of recurrence and of SSMs, and it is why on-device and streaming workloads still use them (§13.3, §16.4). The cost is that the LSTM has only `2 × 4096` floats to remember *anything* with, while the transformer at 8192 tokens holds `8192 × 4096 × 2` floats of K and V. **The transformer buys recall with memory; the RNN buys constant memory with a compression bottleneck.** Neither is free, and which one you want depends entirely on whether your task is recall-bound or latency/memory-bound.

---

## 5. The End-to-End Pipeline

### 5.1 The LSTM era pipeline (what the notebook actually implements)

| Stage | Input → operation → output | Failure mode |
|---|---|---|
| **1. Vocabulary** *(per task, per corpus, from scratch)* | 25 000 raw reviews → `imdb.load_data(num_words=10000)` → ints in `[1, 9999]`, offset `+3` (PAD=0, START=1, UNK=2, UNUSED=3) | Index mapping is corpus-specific. Change corpus → every index is garbage |
| **2. Padding** | Variable lengths (review 1 = 280 tokens, review 2 = 181) → `pad_sequences(maxlen=200, padding='post', truncating='post')` → `(N, 200)` | Post-truncation silently drops content past token 200. IMDb reviews average ~230 tokens, so most lose their verdict sentence |
| **3. Subset** | 25 000 → `x_train[:3000]` → 3 000 rows | 12 % of the data the task ships with. The "tiny dataset" problem, self-inflicted |
| **4. Model** | `Input(200)` → `Embedding(10000, 128)` → `LSTM(256, return_state=True)` → `[state_h]` → `Dense(1, sigmoid)` → 1 674 497 trainable params | The LSTM unrolls 200 steps → 200 sequential kernel launches per batch (§4.4) |
| **5. Train** | `compile(adam, binary_crossentropy, accuracy)`; `fit(batch_size=64, epochs=1, validation_split=0.1)` → 89 s, val_acc 0.5180 | 1 epoch on 3 000 rows = **47 optimizer steps**. Underfit, not converged |
| **6. Predict + decode** | `predict(x[0])` → 0.4902 → *"Negative 😞"*; `reverse_word_index = {i+3: w}` → decoded text | 0.4902 is a coin flip. The printed label is noise |
| **7. Save** | `classification_model.save("lstm_imdb_model.h5")` | HDF5 is legacy (Keras warns); the artifact carries no config, no tokenizer, no vocab file, no metrics |

### 5.2 The "fine-tuning" attempt — and why it is not fine-tuning

| Stage | What it does | Why it fails |
|---|---|---|
| **8. "Retrain"** *(not fine-tuning)* | `load_model(...)`; `compile(adam, binary_crossentropy)`; `fit(x[:1000], y[:1000], batch_size=64, epochs=1)` with `vocab_size` now 1000 → val_acc 0.6000, val_loss 0.6835 | (a) every token ranked 1000–10000 now maps to index 2 (`<UNK>`), so the frozen embedding is fed a distribution it never saw (§6.5); (b) 1 000 rows / 64 = **15 optimizer steps** |
| **9. "Retrain for a different task"** | Reuse `layers[1]` (Embedding) + `layers[2]` (LSTM) as an encoder; build a decoder (`output_vocab_size=8000`) from scratch; `Model([enc_in, dec_in], dec_out)`; `decoder_input_data = np.random.randint(...)`; `decoder_target_data = np.random.randint(...)`; `fit` → loss 8.9872, acc 1.2756e-04 | The targets are **uniform noise**. The only achievable loss is `ln(8000) = 8.9872` and the model achieves it exactly. See §6.5 — the clearest possible accidental demonstration of why the approach failed |

**The five reasons the same weights cannot be repurposed** — the instructor lists them and every one is correct [06:43:32–46:22]:

| # | Reason | His words | Concrete manifestation in the notebook |
|---|---|---|---|
| 1 | **Task-type mismatch** | "you have seen the task type mismatch — first task basically it was a classification task and the next task basically was a summarization" | `Dense(1, sigmoid)` + `binary_crossentropy` vs `Dense(8000, softmax)` + `sparse_categorical_crossentropy` |
| 2 | **Architecture mismatch** | "for this particular task the architectural also is a mismatch… many to one [vs] many to many" | The many-to-one model discards all `y_t` except `y_T`; the summarizer needs `return_sequences=True` on both sides plus a decoder |
| 3 | **Vocabulary differences** | "if the vocabulary is going to be different in that case the encoder part is also going to be fail" | The retrain cell drops `num_words` from 10000 to 1000 — a 10× vocabulary shrink feeding a frozen embedding table |
| 4 | **Out-of-vocabulary** | "out of vocabulary issue means we haven't trained it for huge amount of data… our encoder was trained on IMDb data set but anything is coming from any other data set while we are doing testing it might fail" | No `<UNK>` handling strategy, no subword units, no shared vocab file |
| 5 | **Objective-function mismatch** | "for the classification there is a category binary cross entropy… but for the summarization there is a cross entropy" | `binary_crossentropy` on a scalar vs `sparse_categorical_crossentropy` on `(batch, 50, 8000)` |

He adds the honest conclusion: "what can be reused from this particular model… we can only use the encoder weights **but again it is not a prominent solution** for the summarization task" [06:45:53–46:02].

### 5.3 What the transformer-era pipeline replaced it with

```
STAGE 1  VOCABULARY   shared BPE tokenizer, 32K-128K merges, fixed across all tasks
                      (reused verbatim by every downstream fine-tune)
STAGE 2  PACKING      concatenate + chunk to max_len (no [PAD] waste); block-diagonal mask
STAGE 3  DATA         the whole corpus, not 12% of it; 1M-15T tokens
STAGE 4  MODEL        one architecture (decoder-only transformer), config-scaled
STAGE 5  PRETRAIN     next-token prediction; loss is the *same* loss every downstream
                      task will be cast into
STAGE 6  FINE-TUNE    load the checkpoint, swap the head or write a prompt;
                      LoRA/QLoRA touches 0.5-2% of parameters
STAGE 7  SAVE         safetensors + config.json + tokenizer files + a model card
```

**The single structural change that made this possible is stage 5.** In the LSTM era, each task had its own architecture, its own loss, and its own vocab. In the transformer era, *every task is cast into the same loss* — next-token prediction over a shared vocabulary. Classification is "generate the label token". Summarization is "generate the summary". The instructor reaches exactly this insight: "see fundamental everything is a generation only… in classification also one word is being generated, summarization also some summarize sentence being generated, question answer normal answer is being generated, translation some translation part is being generated… fundamentally everything is a generated one but the variety of the task was different and one model, the LSTM model, was not able to handle this thing" [06:48:47–49:14].

That is the thesis of the module stated correctly by the instructor, and it is worth noting that he arrives at the *right* conclusion (unified loss enables one model) while attributing it to the *wrong* mechanism (LSTM "could not handle it" as a capacity claim, when it is a throughput claim). ULMFiT proved an LSTM *could* handle multiple tasks with a unified pretraining loss — it just could not be scaled.

---

## 6. Hands-On Code (annotated)

Every block below is the companion notebook, cleaned up, with the *why*. Library versions the notebook targets: **TensorFlow 2.16+ / Keras 3** (the `keras.src.callbacks.history.History` repr and the "Optimizer params" line in `summary()` are Keras 3 signals), NumPy 1.26+.

### 6.1 Imports and configuration

```python
# ---- notebook cell 1-2 (verbatim imports, annotated) --------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model                      # functional API
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Why the functional API and not Sequential? Because the seq2seq section later
# needs a model with TWO inputs (encoder_inputs, decoder_inputs). Sequential
# cannot express that. Use functional from the start.

# ---- notebook cell 2 (hyperparameters) ---------------------------------------
max_len       = 200     # tokens per review after padding/truncation
vocab_size    = 10000   # top-10k words from the IMDb frequency ranking
embedding_dim = 128     # words -> 128-d vectors
latent_dim    = 256     # LSTM hidden width (d_h); also state_h/state_c width
```

| Param | Value | Note |
|---|---|---|
| `max_len` | 200 | IMDb reviews average ~230 whitespace tokens → most reviews are truncated. `padding='post'` puts zeros at the end, which matters because a *pre*-padding scheme shifts every real token's position |
| `vocab_size` | 10000 | 10 K words is roughly where the IMDb frequency curve flattens; going to 30 K saves ~1 % OOV at the cost of 2.56 M extra embedding params |
| `embedding_dim` | 128 | 128 is small for 10 K words. Modern practice: 256–768 for a from-scratch embedding on a 10 K vocab |
| `latent_dim` | 256 | Determines both the state size and the parameter count of the recurrence |

### 6.2 Data loading, subsetting, padding

```python
# ---- notebook cell 3-7 --------------------------------------------------------
(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
# num_words=10000 keeps the 10,000 most frequent words. Everything else in the
# corpus is replaced by index 2 = <UNK>. The `_` discards the 25,000-row test set
# -- fine for a demo, wrong for any real evaluation (see §12).

assert len(x_train) == len(y_train) == 25000

# The instructor truncates to 3,000 rows "just to train my model" [06:26:35-26:42]
x_train = x_train[:3000]
y_train = y_train[:3000]
# WARNING: this is a 12% subset of a dataset that is itself small by 2026
# standards, and it is not shuffled. IMDb is ordered with negatives first in some
# Keras versions, so x_train[:3000] may be class-skewed. ALWAYS shuffle before
# subsetting:  idx = np.random.permutation(len(x_train)); x_train = x_train[idx][:3000]
# The notebook's reported val_accuracy of 0.5180 is consistent with a near-balanced
# subset, so the ordering happened not to hurt here -- but it is luck, not design.

# Pad/truncate to a fixed width so the LSTM can be unrolled on a static graph
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
# padding='post'    -> zeros appended at the END
# truncating='post' -> drop tokens from the END when len > 200
# For sentiment classification, a head+tail truncation (first 100 + last 100 tokens)
# is measurably better than head-only or tail-only, because IMDb reviews put the
# verdict in the first and last few sentences. Keras' pad_sequences cannot do this;
# you have to do it manually before padding.
```

> **Beyond the video — the padding/truncation choice is worth 1–3 accuracy points and nobody talks about it.** The instructor's own decode cell (6.4) shows the problem: the decoded review ends mid-sentence at `"...don't you think the whole story was"`. The verdict sentence of that review is *past* token 200 and has been thrown away. For IMDb specifically, head+tail truncation at 200 tokens is a documented improvement over head-only. For a real classifier, use dynamic padding (pad each batch to its own max) so that a 60-token review is not spending 140 positions on `<PAD>`.

### 6.3 The classification model

```python
# ---- notebook cell 8 ----------------------------------------------------------
input_layer     = Input(shape=(max_len,))                       # (None, 200) int32
embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)   # (None, 200, 128)

# return_state=True is THE key flag. It makes the LSTM return three tensors
# instead of one: the full output sequence, the final hidden state, the final
# cell state. We only want state_h (and, later, state_c).
lstm_layer, state_h, state_c = LSTM(latent_dim, return_state=True)(embedding_layer)
# lstm_layer : (None, 200, 256)  -> DISCARDED here (many-to-one)
# state_h    : (None, 256)       -> the compressed summary of the sequence
# state_c    : (None, 256)       -> the cell state; unused here, needed later

output_layer = Dense(1, activation='sigmoid')(state_h)          # (None, 1) in (0,1)
classification_model = Model(input_layer, output_layer)
```

**This is the many-to-one pattern, and the `state_h, state_c = ...` unpacking is the whole reason the retrain section can build a decoder later.** If you write `LSTM(256)` without `return_state=True`, you get only `lstm_layer` and there is no way to recover the initial state for a decoder.

Verify the parameter arithmetic against the notebook's own summary:

```python
classification_model.summary()
```

| Layer | Output shape | Params | Derivation |
|---|---|---|---|
| `input_layer` | `(None, 200)` | 0 | |
| `embedding` | `(None, 200, 128)` | **1,280,000** | `10000 × 128` |
| `lstm` | `[(None,256), (None,256), (None,256)]` | **394,240** | `4 × (128×256 + 256×256 + 256) = 4 × 98560` |
| `dense` | `(None, 1)` | **257** | `256 × 1 + 1` |
| **Trainable params** | | **1,674,497** | = 1,280,000 + 394,240 + 257 ✓ |
| Non-trainable | | 0 | |
| Optimizer params | | 3,348,996 | Adam's `m` and `v`, 2 × 1,674,497 ✓ |
| **"Total params"** | | **5,023,493** | Keras 3 folds optimizer state into "total" |

> **Beyond the video — read that summary correctly.** Keras 3's `Total params: 5,023,493 (19.16 MB)` is **not** the model size. The model is **1,674,497 params ≈ 6.39 MB in fp32** (which the summary itself prints as `Trainable params: 1,674,497 (6.39 MB)`). The extra 3.35 M are Adam's two moment buffers, which exist only during training and are not part of the artifact. If you quote 19 MB for this model in an interview you are off by 3×. **The rule: model size = trainable + non-trainable. Optimizer state is a training cost, not a model cost.**

### 6.4 Compile, train, predict, decode

```python
# ---- notebook cell 11-14 ------------------------------------------------------
classification_model.compile(optimizer='adam',
                             loss='binary_crossentropy',   # sigmoid + BCE, correct pairing
                             metrics=['accuracy'])

history = classification_model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=1,                 # <-- ONE epoch
    validation_split=0.1,     # 10% held out, not 0.1% (the video says "0.1%" [06:30:12])
)
# 71/71 - 89s 1s/step - accuracy: 0.5011 - loss: 0.6942 - val_accuracy: 0.5180 - val_loss: 0.6926
```

**Read those numbers.** `loss = 0.6942` against `−ln(0.5) = 0.6931`. The model is emitting ≈ 0.5 for every input. `accuracy = 0.5011` is a coin flip. `val_accuracy = 0.5180` is a coin flip with a 30-sample swing (300 val rows; the 1σ binomial noise is `√(0.25/300) = 2.9 %`).

The training budget explains it: `3000 rows × 0.9 / 64 = 42` optimizer steps. **Forty-two gradient steps.** A model this size needs thousands. This is underfitting, and the video presents it as a working classifier.

```python
# Inference on one review
sample_review = x_train[0].reshape(1, -1)      # (1, 200)
prediction = classification_model.predict(sample_review)
# Predicted sentiment probability (positive): 0.4902157      <- 0.49 = coin flip
print("Predicted Sentiment:", "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞")
```

The `> 0.5` threshold on an untrained model produces a *label* that carries no information. **Production rule: never expose a hard label from a model whose calibration you have not measured.** Emit the probability, and gate on a calibrated threshold.

```python
# ---- notebook cell 14: decode integer IDs back to words ----------------------
word_index = imdb.get_word_index()

# The +3 offset. Keras reserves indices 0,1,2 for PAD/START/UNK (the sequences
# returned by load_data are already shifted by 3 relative to get_word_index()).
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

decoded_review = " ".join(reverse_word_index.get(i, "<UNK>") for i in sample_review[0])
print(decoded_review)
```

Output (abridged): `<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played … don't you think the whole story was`

Three things to notice:
1. The review is **positive** and the model called it Negative at 0.49. Confirms the model is untrained.
2. The text **ends mid-sentence** — proof of the token-200 truncation in §6.2.
3. `imdb.get_word_index()` is called but `imdb` was **never imported** (`tf.keras.datasets.imdb.load_data` was used). The video hits exactly this error on camera: "IMDb is not defined, uh let me check whether I have defined the IMDb or not" [06:31:44–32:21]. He fixes it by importing `imdb` from the datasets module. The cleaned-up version:

```python
from tensorflow.keras.datasets import imdb   # <- the line the notebook was missing
```

### 6.5 The retrain / "fine-tune" attempt — and the two silent failures

```python
# ---- notebook cells 17-22 ------------------------------------------------------
max_len, embedding_dim, latent_dim = 200, 128, 256
vocab_size = 1000                                    # <-- CHANGED from 10000
(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_train, y_train = x_train[:1000], y_train[:1000]    # <-- 15 optimizer steps at bs=64

load_classification_model = load_model("lstm_imdb_model.h5")
load_classification_model.compile(optimizer='adam', loss='binary_crossentropy',
                                  metrics=['accuracy'])
load_classification_model.fit(x_train, y_train, batch_size=64, epochs=1,
                              validation_split=0.1)
# 15/15 - 22s 1s/step - accuracy: 0.5386 - loss: 0.6876 - val_accuracy: 0.6000 - val_loss: 0.6835
load_classification_model.save("lstm_imdb_model_updated.h5")
```

**Silent failure #1 — the vocabulary changed under the embedding.** `load_data(num_words=1000)` keeps the top 1000 words and maps everything else to index 2 (`<UNK>`). But the embedding table has 10 000 rows and was trained on a distribution where index 2 was rare. Now index 2 is *most tokens*, and the encoder is being fed a distribution it never saw. `val_accuracy 0.6000` on 100 validation rows means ±10 % — the number is noise.

> **Beyond the video — this is the concrete, mechanical version of the instructor's abstract "vocabulary differences" and "out of vocabulary" points [06:44:23–45:26].** Note that the *index mapping is stable* for indices below 1000 (Keras assigns indices by frequency rank, and the ranking does not change when you lower `num_words`). So this is not a "the indices mean different words" bug — it is worse, because it is invisible: the indices are correct, the embedding rows are correct, and yet the token *distribution* has been replaced with one where a single token (`<UNK>`) dominates. The model has no way to signal that something is wrong. This is the archetype of a silent failure mode: **no exception, no shape error, a plausible-looking accuracy, and a model that has been fed out-of-distribution inputs.**

**Silent failure #2 — the seq2seq section trains on noise.**

```python
# ---- notebook cells 24-29: "Retraining for Summarization" ---------------------
updated_classification_model = load_model("lstm_imdb_model_updated.h5")

# Reuse the pretrained encoder layers by INDEX.
#   layers[0] = InputLayer
#   layers[1] = Embedding        <- reused
#   layers[2] = LSTM             <- reused, and its (state_h, state_c) initialise the decoder
#   layers[3] = Dense            <- abandoned
encoder_inputs  = Input(shape=(max_len,))
encoder_embedding = updated_classification_model.layers[1](encoder_inputs)
encoder_outputs, state_h, state_c = updated_classification_model.layers[2](encoder_embedding)

# Decoder must be built from scratch: the classification model had no decoder.
output_vocab_size = 8000
target_max_len    = 50
decoder_inputs  = Input(shape=(None,))
decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm    = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding,
                                     initial_state=[state_h, state_c])   # <- THE BOTTLENECK
decoder_dense   = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
seq2seq_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

encoder_input_data  = x_train[:1000]                                       # real IMDb
decoder_input_data  = np.random.randint(1, output_vocab_size, (1000, 50))  # RANDOM
decoder_target_data = np.random.randint(1, output_vocab_size, (1000, 50, 1))# RANDOM

seq2seq_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=32, epochs=1, validation_split=0.1)
# 29/29 - 67s 2s/step - accuracy: 1.2756e-04 - loss: 8.9872 - val_accuracy: 0.0 - val_loss: 8.9875
```

**Derive the loss.** A softmax classifier over `V = 8000` classes that emits the *uniform* distribution has cross-entropy

```
H_uniform = −Σ_{v=1}^{V} (1/V) · ln(1/V) = ln(V) = ln(8000)
          = ln(8) + ln(1000) = 2.0794415 + 6.9077553
          = 8.9871968…
```

The notebook reports **`loss: 8.9872`** and `val_loss: 8.9875`, and `accuracy: 1.2756e-04`. `1.2756e-04` is, within noise, exactly `1/8000 = 1.25e-04`. **The model outputs a uniform distribution over 8000 tokens and gets the argmax right only by chance.** The targets were drawn from `np.random.randint`, which is uniform. The model has *exactly* learned the target distribution and nothing else, and it is the global optimum for that data.

Four independent errors in this cell, all of which must be fixed before it means anything:

| # | What is wrong | Why it matters | The fix |
|---|---|---|---|
| 1 | `decoder_target_data` is random noise, not summaries | There is no signal to learn; ln(V) is the floor | Use a real summarization corpus (CNN/DailyMail, XSum) with `(article, summary)` pairs |
| 2 | `decoder_input_data` is random, not the target shifted right | This is **not teacher forcing**, despite the instructor explaining teacher forcing correctly in the video [06:39:08–40:00] | `dec_in = target[:, :-1]`, `dec_target = target[:, 1:]` |
| 3 | The encoder's last hidden state is the *only* channel from source to decoder | The information bottleneck of §4.3.4(b) — the exact failure Bahdanau attention was invented to fix | Add attention: `Attention()` over `encoder_outputs` (needs `return_sequences=True` on the encoder LSTM) |
| 4 | `output_vocab_size = 8000` is unrelated to the encoder's `vocab_size = 1000` | Source and target share no vocabulary space, so no weight tying and no shared semantics is possible | Use one tokenizer for both, with proper `<sos>`/`<eos>` |

> **The honest reading of this notebook.** It is not a demonstration that "LSTM cannot do summarization." It is a demonstration of what fine-tuning looked like in the RNN era: you rebuilt the architecture per task, you had no shared vocabulary, you had no shared objective, you had no framework support, and a cell that trains on random noise produces a number that looks like a loss. **The three things a modern practitioner would flag in 30 seconds — no teacher forcing, random targets, no attention — are exactly the three things the transformer era standardized away.** That is the module's thesis, demonstrated accidentally by the companion notebook itself.

### 6.6 What to change for your own data — the checklist

| If you are doing this | Change this | Because |
|---|---|---|
| Real sentiment/classification on your own corpus | Replace `imdb.load_data` with a `TextVectorization` layer or a Hugging Face tokenizer; fit the vocab **on your training split only** | Fitting on the full corpus leaks val/test vocabulary into the model |
| Any sequence longer than ~100 tokens | Reduce `max_len`, or switch to TBPTT with `stateful=True` and `k=35–100` | Full BPTT over 200 steps is where the vanishing gradient bites hardest |
| Any seq2seq task | Add attention over `encoder_outputs`; implement real teacher forcing; use one shared tokenizer | The `(h,C)` bottleneck is the single largest source of quality loss |
| Deployment | Save `model.keras` (not `.h5`) **plus** the tokenizer config, the vocab, the label map, and metrics | HDF5 is legacy and carries no preprocessing provenance |
| Anything at all in 2026 | Do not do this. Use a pretrained transformer + LoRA (CS-13 §6.8, CS-11 §4.11) | The whole point of this module |

---

## 7. Hyperparameters & Configuration — Every Knob

| Param | What it does | Notebook value | Safe range | Too high → | Too low → | Framework flag |
|---|---|---|---|---|---|---|
| `max_len` | BPTT depth; sequence truncation point | 200 | 32–512 for LSTM | Longer gradient path, `O(T·d²)` per layer, memory grows linearly | Truncates away the signal; for IMDb, the verdict sentence | `pad_sequences(maxlen=)` |
| `latent_dim` / `d_h` | Recurrent state width | 256 | 128–1024 | Quadratic params (`4d²`), slower per step, more overfitting | Underfitting; state is an even tighter bottleneck | `LSTM(units=)` |
| `embedding_dim` | Word vector width | 128 | 128–768 | Embedding table dominates param count at large vocab | Vectors cannot separate word senses | `Embedding(_, output_dim)` |
| `vocab_size` | Embedding table rows | 10000 | 8 K–50 K (word-level) / 32 K–128 K (BPE) | Rare words have too few gradient updates to learn | High OOV rate | `num_words=` / tokenizer `vocab_size` |
| `batch_size` | Rows per optimizer step | 64 | 32–256 | Fewer steps per epoch → underfitting in short runs; more VRAM | Noisy gradients, poor throughput on tensor cores | `fit(batch_size=)` |
| `epochs` | Passes over data | **1** | 3–30 for from-scratch LSTM | Overfitting an untrained-from-scratch embedding | **Underfitting — this is the notebook's bug** | `fit(epochs=)` |
| `validation_split` | Fraction held out *from the training tensor* | 0.1 | Prefer an explicit val set | Too little training data | Noisy val metric (±2.9 % at 300 rows) | `fit(validation_split=)` |
| `optimizer` | Update rule | adam | adam / adamw | lr too high → loss spikes | lr too low → 42 steps does nothing | `compile(optimizer=)` |
| `learning_rate` | Step size | 1e-3 (Keras default) | 1e-4 – 3e-3 for LSTM | Divergence, NaN in a few hundred steps | Converges too slowly to see in 1 epoch | `Adam(learning_rate=)` |
| `clipnorm` | Global-norm gradient clipping | **not set — Keras default is 1e9, i.e. off** | 0.25–1.0 for RNN/LSTM | Exploding updates, spiking loss | Clipped gradient is too weak; slow learning | `Adam(clipnorm=1.0)` |
| `return_state` | Emit final `(h, C)` | `True` | `True` whenever a decoder follows | — | Cannot build a seq2seq decoder | `LSTM(return_state=)` |
| `return_sequences` | Emit all `y_t` | `False` (encoder) / `True` (decoder) | — | Unnecessary memory and compute on a many-to-one head | `Dense` on a 3-D tensor fails, or attention has nothing to attend to | `LSTM(return_sequences=)` |
| `padding` | Where zeros go | `'post'` | `'post'` | — | `'pre'` shifts every token's index and wastes the LSTM's early steps on pads | `pad_sequences(padding=)` |
| `truncating` | Which end to drop | `'post'` | `'post'` + manual head+tail for classification | — | `'pre'` drops the review's opening context | `pad_sequences(truncating=)` |
| `dropout` | Regularization | not set | 0.1–0.5 recurrent, 0.1–0.3 embedding | Slow convergence, underfitting on small data | Overfitting on 3 000 rows | `LSTM(dropout=, recurrent_dropout=)` |
| `recurrent_dropout` | Dropout on the recurrent connection | not set | 0–0.3 | **Disables the cuDNN fused kernel → 5–10× slower** | Default 0 is fine | `LSTM(recurrent_dropout=)` |
| `unroll` | Statically unroll the loop | not set | `True` only for short fixed `T` | Large graph, slow compile | — | `LSTM(unroll=)` |
| `stateful` | Carry state across batches | not set | Only with a documented TBPTT scheme | Silent correctness bugs if you forget `reset_states()` | Chunked-context training impossible | `LSTM(stateful=)` |

### 7.1 The three knobs that actually decide whether it trains

**`clipnorm` (or `clipvalue`) is the highest-leverage setting and it is off by default.** Keras' `Adam` defaults to `clipnorm=None` (internally `1e9`) and PyTorch's `clip_grad_norm_` defaults to `max_norm=1e9`. In the RNN era, every recipe had `clipnorm=1.0`. Two rules:
- Use `clipnorm` (global norm), never `clipvalue`. `clipvalue` clips each component independently, which rotates the gradient direction — it is a different and usually worse optimization problem.
- Log the pre-clip gradient norm. If it is consistently 10× the clip value, the model is in a regime where clipping is masking a structural problem (learning rate too high, init too large), not fixing it.

**`recurrent_dropout` is a performance trap.** Setting it to any nonzero value forces Keras off the cuDNN fused RNN kernel onto a Python loop over time steps. Measured cost on a T4 for a 2-layer LSTM with `T=200`, `d_h=512`: **~8× slower end to end**. If you need recurrent regularization, use `dropout` (on the input/output, kernel-compatible) or `zoneout` implemented manually, or accept the cost knowingly.

**`stateful` + TBPTT is the only way to train on long documents** and it is a correctness minefield. In Keras you must (a) set `batch_size` explicitly on the first layer, (b) shuffle *by document chunk* and not across documents, and (c) call `reset_states()` at document boundaries. Get any of those wrong and the model silently trains on a state it should have discarded — a failure that shows up as a slightly-worse-than-expected loss and nothing else.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 Which architecture, given a real problem in 2026

| Situation | Use the transformer? | Instead use | Why |
|---|---|---|---|
| Text classification, NER, QA, summarization at any scale | **Yes, always** | — | A pretrained checkpoint beats a from-scratch LSTM by 5–25 points on every one of these, for 1 % of the compute |
| Streaming audio, keyword spotting, always-on wake word | No | Small RNN/LSTM or DS-CNN, TFLite Micro | Constant state, no KV cache, runs in 100 KB of RAM |
| Inference on a microcontroller (≤1 MB RAM) | No | 1–4 layer LSTM/GRU, INT8 | A KV cache for even 512 tokens does not fit; an LSTM state is `2×d_h` bytes |
| Very long sequences where memory is the binding constraint | Hybrid or SSM | Mamba-2, Jamba, Zamba, Bamba | Constant-size state vs `O(n)` KV cache. §16.3 |
| Training data < 500 labeled examples, task is text | No | A prompted pretrained LLM + few-shot, or a frozen-embedding + logistic regression | From-scratch LSTM needs thousands; the pretrained model needs tens |
| Time series forecasting, small data, strong temporal prior | Sometimes | LSTM/GRU, or a specialized model (N-BEATS, PatchTST, TimesNet) | Transformers underperform on short, low-dimensional, noisy series without heavy tuning |
| RL world models / model-based RL | No | RSSM (DreamerV3), GRU-based | Recurrent belief states are the natural formulation and the sequences are short |
| Anything requiring unbounded exact recall (copy, lookup, retrieval over the input) | Yes | Transformer | A fixed-size recurrent state provably cannot do unbounded associative recall |
| Real-time control loop with a <1 ms budget per token | Sometimes | Small LSTM, or a distilled transformer | A 4-layer LSTM at `d=256` is ~1 M FLOPs/token; a 7B transformer is not |

### 8.2 STOP conditions — signals you are using the wrong tool

1. **You are about to train a sequence model from scratch and you have a pretrained checkpoint that fits the modality.** Stop. The checkpoint wins. This was true from 2018 onward and is not close in 2026.
2. **Your sequence length is `> 3 × d_model` and you are worried about attention's `O(n²)`.** Stop and check the FFN first — below that crossover the FFN dominates. §16.2 has the arithmetic.
3. **You are choosing an LSTM because "it has a smaller memory footprint."** Stop and compute the actual numbers. The LSTM's advantage is *constant* memory, which matters only when `n` is large or the memory budget is tiny. At `n = 512` with `d_h = 512`, an LSTM state is 4 KB/layer while a KV cache is 0.5 MB/token × 512 = 256 MB total — the LSTM wins by 1000×. At `n = 8` the same comparison is 4 KB vs 4 MB — still wins, but nothing cares.
4. **You are adding `recurrent_dropout` to fix overfitting.** Stop. Use `dropout` or reduce `latent_dim`. You just cost yourself 8× wall clock.
5. **Your validation accuracy is 50 % after one epoch and you are about to tune the learning rate.** Stop. Count your optimizer steps. `rows × (1 − val_split) / batch_size` is the only number that matters. Under 500 steps, no learning rate will save you.
6. **You are going to fine-tune an LSTM checkpoint.** Stop. There is no such thing to fine-tune — no public checkpoint, no shared tokenizer, no framework support. This is the module's thesis.

### 8.3 The single-question screen

> *"Can you name the pretrained checkpoint you are starting from, and is it the same architecture your tooling was built for?"*

If the answer is no for either half, you are in the LSTM era, and the rest of this module explains why that era ended.

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros — what recurrence is genuinely good at

| Pro | Mechanism | Where it still wins |
|---|---|---|
| `O(1)` inference state per token | `h_t` replaces `h_{t−1}`; nothing accumulates | Streaming, long-running sessions, constant-RAM serving |
| Constant inference memory in `n` | State size is `2·d_h` regardless of sequence length | 1M-token streaming with a 512 KB budget |
| Linear compute in `n` | `O(n·d²)` FLOPs, no `n²` term | Very long sequences where the `n²` term dominates |
| Natural variable-length handling | No positional encoding needed; nothing to truncate | Sensor streams with irregular sampling |
| Strong inductive bias for local structure | The recurrence is a smoothness prior over time | Low-data time series, control, audio |
| Trivially causal | The architecture is causal by construction; no mask to get wrong | Autoregressive generation with zero masking bugs |
| Trainable on a CPU | ~1 M FLOPs/token for a 4-layer GRU at `d=256` | Edge deployment, embedded inference |

### 9.2 Cons

| Con | Mechanism | Cost |
|---|---|---|
| No parallelism across time | `h_t` depends on `h_{t−1}` | ~100× lower GPU throughput at equal parameter count (§4.4.2) |
| Path length `O(n)` between tokens | Two tokens `k` apart interact through `k` nonlinear layers | Gradient decays as `λ^k`; effective dependency ≈ 100–200 steps |
| Fixed-size state bottleneck | All history compressed into `2·d_h` floats | Encoder–decoder BLEU collapses with length |
| Per-task architecture | Many-to-one ≠ many-to-many ≠ encoder–decoder | No shared artifact, no transfer, no tooling |
| Per-task vocabulary | Word-level vocab is corpus-specific | OOV on any new domain; embedding table is dead weight |
| Requires gradient clipping | Exploding gradients | A silently-missing config line produces a diverged run |
| `O(n)` serial kernel launches | One launch per time step | Launch overhead dominates at small `d_h` |
| No public pretrained checkpoints | Never standardized | You pay 100 % of the training cost every time |

### 9.3 Hard limitations — not fixable by tuning

1. **You cannot make an RNN parallel in time.** Every workaround (quasi-RNN, SRU, diagonal RNNs) either loses expressivity or reintroduces a sequential dependency. The only genuine fix is to remove the recurrence (attention) or to make the recurrence *linear and time-invariant* so that it becomes a convolution (S4/Mamba-1 without selectivity).
2. **You cannot exceed the state capacity.** A `d`-dimensional real state cannot store more than `~d` independent real values' worth of information. This is not a training issue; it is linear algebra. It is the reason SSMs and RNNs lose on associative-recall tasks (Jelassi et al. 2024, *Repeat After Me*).
3. **You cannot recover a dead gradient.** Once `λ^k` is below fp32 epsilon for the paths that matter, no optimizer configuration brings those paths back. Adam's scale invariance does not help (§4.2.3).
4. **You cannot transfer a word-level embedding across vocabularies.** Index `4213` means "brilliant" in your IMDb vocab and something else entirely in another. That is a hard blocker for transfer, not a hyperparameter.

### 9.4 Silent failure modes — looks fine, is broken

| Symptom | What is actually happening | How to detect in 5 minutes |
|---|---|---|
| `val_accuracy ≈ 0.5`, `loss ≈ 0.693` | Model emits ≈ 0.5 everywhere; underfit or dead gradient | Compare loss to `−ln(1/num_classes)`. If they match, the model learned nothing |
| `loss = ln(V)` exactly on a seq2seq task | Targets are uniform noise, or the head is untrained and the loss is the uniform floor | Compute `ln(V)` and compare to 3 decimals (§6.5) |
| Accuracy looks okay but the confusion matrix is degenerate | Predicting the majority class | Always print `confusion_matrix`, not just accuracy |
| Val accuracy above train accuracy and both mediocre | `validation_split` took a non-random subset, or the model is trivially fit | Shuffle before splitting; check the class balance of the split |
| Loss decreases smoothly but generation is incoherent | Teacher forcing at train time, free-running at inference — exposure bias | Run a free-running eval; if it collapses immediately, that is the gap |
| Training is 8× slower than a benchmark | `recurrent_dropout > 0` disabled the cuDNN kernel | Set it to 0 and re-time |
| NaN loss at step ~200 | Exploding gradient, no clipping | `clipnorm=1.0`; also log the pre-clip grad norm |
| "Fine-tuned" model is worse than the base on the original task | Catastrophic forgetting from full-weight updates on a small dataset | Evaluate on a held-out sample of the *original* distribution before and after |
| Model works on inputs of length ≤ 200 and degrades sharply above | Trained with `max_len=200` and no positional generalization (RNN: TBPTT cap; transformer: position embeddings) | Evaluate by length bucket; the degradation curve's knee is your real limit |

---

## 10. Corrections, Exceptions & Gotchas

### 10.1 `Correction:` "RNN remembers 10–15 words, LSTM 30 words"

**The instructor says** [06:15:35–15:54, repeated at 07:20:13–20:18]: "in RNN there was a short-term memory means we cannot remember like more than 10 to 15 words… but in LSTM basically we can recall that sentence up to 30 words."

**What is right:** the *ordering* (RNN < LSTM) and the *existence* of a length limit are both correct.

**What is wrong:** the specific numbers are a folk heuristic with no experiment behind them, and they are wrong in both directions depending on the task.

**The correct statement:**

| Claim | Evidence | Number |
|---|---|---|
| A simple RNN's gradient decays exponentially in the gap | Bengio, Simard & Frasconi (1994), the original proof | Signal is unusable beyond ~**10–20** steps |
| An LSTM's *effective* context in a trained LM is much longer | Khandelwal et al. (2018), *Sharp Nearby, Fuzzy Far Away* — a gate-based probe of a 2-layer LSTM LM | Sharp window ~**50** tokens; usable context to ~**200** tokens |
| An LSTM can *learn* a dependency of 1000+ steps | Hochreiter & Schmidhuber's original 1997 constant-error-carousel tasks | **1000+** steps, with a carefully initialized forget gate |
| The practical limit is throughput, not memory | AWD-LSTM (Merity et al. 2017), 57.3 perplexity on Penn Treebank, single GPU | ~**24 h** per run — the training cost, not the memory, is what caps it |

**Interview-safe phrasing:** *"Distance-in-steps is the variable that matters, and the decay is exponential in it: `λ^n`. A simple RNN's `λ` is set by `σ_max(W_hh)·tanh'`, which is near 1 by construction at init but drifts, and it degrades past 10–20 steps. An LSTM's `λ` on the cell path is `f_t`, which is learnable and bounded, so a tuned LSTM carries signal to ~100–200 steps. The 10–15 / 30 numbers you see in blog posts are not measurements."*

### 10.2 `Correction:` the residual connection is addition, not concatenation

**The instructor says** [07:31:11–31:28]: "whatever uh input we are getting from here… as it is we are passing it over here and whatever thing we are going to be processed through the multi attention both thing we are going to be **concatenating**… then we are passing it to the normalization."

**It is element-wise addition:**

```
z = LN( x + MHA(x) )      not      z = LN( [x ; MHA(x)] )
```

**Why this matters, not just pedantically:** concatenation would grow the width by `d_model` at every block — after 32 layers the tensor would be `33 · d_model` wide, the parameter counts would be completely different, and `∂z/∂x` would be a *block matrix* `[I ; ∂MHA/∂x]` rather than the sum `I + ∂MHA/∂x`. **The sum is what produces the clean identity gradient path.** The identity `∂(x + F(x))/∂x = I + ∂F/∂x` is the entire reason a 100-layer transformer trains: the `I` term carries the gradient to layer 1 at full magnitude regardless of how many `∂F/∂x` terms multiply together. Concatenation gives you no such term.

(The instructor does correctly identify the *purposes* — "regulate the gradient flow… stabilize training and speed up the training" [07:31:32–31:49] — he just misnames the operation. Worth noting because "concatenation" is the single most common error in candidate explanations of residual connections.)

### 10.3 `Correction:` the weight update formula is missing the learning rate

**The instructor writes on screen** [07:11:12–11:39]: `new weight = old weight − dL/dw`.

**The correct update is:**

```
w ← w − η · ∂L/∂w                (SGD)
w ← w − η · m̂ / (√v̂ + ε)        (Adam)
```

The learning rate `η` is not decoration — it is the term that decides whether the update is stable. Dropping it in an interview answer signals that you have memorized a shape without the scale. Also note the sign convention: `−` because you *descend*. Writing `+` is a real error, not a typo.

### 10.4 `Correction:` the forget gate does an element-wise product, not a "cross multiplication"

**The instructor says** [07:17:50–17:58]: "this forget gate actually we are doing a cross the multiplication over here and because of that is going to be forget the information from this cell."

The operation is `f_t ⊙ C_{t−1}`, the **Hadamard (element-wise) product**. It is not a cross product (which is only defined in ℝ³ and produces a vector orthogonal to both inputs) and it is not a matrix product. It is `d_h` independent scalar multiplications, one per memory unit.

**Why the distinction is load-bearing:** the element-wise structure is *precisely* what makes the forget gate a gradient highway. If it were a matrix product, `∂C_t/∂C_{t−1}` would be a full `d_h × d_h` matrix with the same exploding/vanishing dynamics as the RNN. Because it is diagonal, `∂C_t/∂C_{t−1} = diag(f_t)` — a per-unit scalar, bounded in `(0,1)`, with no weight matrix in the path. **The whole LSTM trick is one word: diagonal.**

### 10.5 `Correction:` Mamba is not "an upgraded version of the transformer"

**The instructor says** [06:11:44–11:58]: "we have a mama also… it is a updated uh upgraded version of the transformer… and more than transformer and maybe it could be the future."

**Mamba is a different family, not a version of the transformer.** It is a *selective state-space model* (S6) — a linear recurrence with input-dependent parameters:

```
Continuous:   x'(t) = A x(t) + B u(t),     y(t) = C x(t) + D u(t)
Discretized:  x_k   = Ā x_{k-1} + B̄ u_k,    y_k   = C x_k
```

The lineage matters: S4 (Gu et al. 2021) → S5 → H3 → **Mamba/S6** (Gu & Dao, 2023) → **Mamba-2 / SSD** (Dao & Gu, 2024). Nothing in it is derived from attention. (Mamba-2's *Structured State Space Duality* result shows that a specific SSM is equivalent to a masked attention with a semiseparable mask — a theoretical bridge, not a derivation.)

| | Transformer | Mamba / SSM |
|---|---|---|
| Sequence mixing | `O(n²)` attention | `O(n)` linear recurrence |
| Training parallelism | Full, via GEMM | Full, via a hardware-aware **parallel scan** (Mamba-1) or a matmul form (Mamba-2) |
| Inference state | KV cache, `O(n)` per token | Recurrent state, `O(1)` per token |
| Data-dependent routing | Yes (attention weights) | Mamba-1/2: yes (selective `B, C, Δ`). S4: **no** |
| Long-context recall | Strong | Weak on exact recall; the fixed state cannot hold it |
| 2026 status | Dominant; hybrids are the frontier | Not a drop-in replacement. Used in *hybrid* stacks |

**Correct 2026 statement:** "Mamba is the strongest current member of the state-space family. Pure Mamba matches transformers at small-to-medium scale on language modelling and beats them on throughput, but it does not match them on recall-heavy tasks, which is why the frontier moved to hybrids — Jamba's 1:7 attention-to-Mamba interleave, Zamba, Bamba, Samba, Hymba, Nemotron-H, Qwen3-Next — rather than to pure SSMs." See §16.3.

### 10.6 `Correction:` ULMFiT is 2018, LSTM-based, and it is the *proof* the LSTM era was right about transfer

**The instructor says** [06:46:55–47:26]: "ulm fit that was a research which was introduced along with the transformer itself. There they have used the encoder decoder with the attention… inside the encoder decoder they have used LSTM model. But again… the universal language model fine-tuning paper demonstrated a breakthrough."

**Corrections:**

1. **ULMFiT is 2018 (Howard & Ruder), one year *after* the transformer (2017).** It did not arrive "along with" it.
2. **It is not encoder–decoder and it does not use attention.** It is a **single-stack, 3-layer AWD-LSTM** (24 M parameters, `d = 1152`) trained as a *unidirectional language model*, then fine-tuned with a classifier head. No encoder, no decoder, no attention.
3. **The instructor's underlying point is right and stronger than he states it.** ULMFiT is the paper that proved the pretrain→fine-tune recipe works, and it proved it *on an LSTM*. Its results: **18–24 % error reduction on the majority of datasets**, and — with only **100 labeled examples** — matching a from-scratch model trained on **100× more data**. It also introduced the three techniques everything downstream inherited: **discriminative fine-tuning** (per-layer learning rates), **slanted triangular learning rates** (STLR), and **gradual unfreezing**.
4. **Why it did not become universal, in his own words and correctly:** "LM-based models are computationally inefficient for the larger data. There is a lack of scalability due to the sequential processing" [06:47:39–47:50]. **This is the accurate diagnosis.** The recipe was right; the architecture could not be scaled. BERT and GPT are ULMFiT's recipe with the recurrence removed.
5. **Also correct but garbled:** "later on B GPT5 model which was a transform based model replace LSTM" [06:48:01–48:07]. This should read **BERT (2018) and GPT (2018)** — the transcription captured a mis-speak. There is no "GPT-5" in this history.

### 10.7 `Correction:` the encoder–decoder timeline

**The instructor says** [07:22:43–22:57]: "this is called encoder decoder architecture this was published in 2014 by suska."

Correct, with the completion that an interviewer will expect:

| Year | Paper | Contribution |
|---|---|---|
| 2014 | **Sutskever, Vinyals & Le** — *Sequence to Sequence Learning with Neural Networks* (NeurIPS 2014) | The 4-layer LSTM encoder–decoder; WMT'14 En→Fr **BLEU 34.8**, 36.5 with a 5-model ensemble. **The source of the phrase "seq2seq"** |
| 2014 | **Cho et al.** — *Learning Phrase Representations using RNN Encoder–Decoder* | Introduced the **GRU** and, independently, the encoder–decoder. The instructor's "Suska" is K. **Cho** and I. **Sutskever** conflated |
| 2014/2015 | **Bahdanau, Cho & Bengio** — *Neural Machine Translation by Jointly Learning to Align and Translate* (ICLR 2015) | **Additive attention.** The fix for the bottleneck, and the direct ancestor of the transformer's attention |
| 2017 | **Vaswani et al.** — *Attention Is All You Need* (NeurIPS 2017) | Removed the recurrence; self-attention only. 8× P100, **3.5 days** for base, **12 hours** for big |

**The gap that matters:** attention existed in 2014, three years before the transformer. What the transformer did was **remove the recurrence that attention was wrapped around**. Attention was not the invention; *attention without a recurrence* was.

### 10.8 `Correction:` `validation_split=0.1` is 10 %, not 0.1 %

**The instructor says** [06:30:12–30:15]: "here is the validation is split means 0.1 % data is going to be used for the validation."

Keras' `validation_split` is a **fraction**, so `0.1` = **10 %**. With 3 000 rows that is 300 validation rows, not 3. (If it really were 0.1 % you would have 3 validation rows, and a single misclassification would move the accuracy by 33 %.) This matters for reading the headline metric: `val_accuracy = 0.5180` on 300 rows carries a ±2.9 % binomial standard error, so 0.5180 is statistically indistinguishable from 0.49 or 0.55.

### 10.9 Other gotchas

1. **The notebook's `len(x_train)` cell prints `5000` after the slice to 3 000.** Cells 5 and 6 are supposed to show 3 000. The `5000` shown in the saved output is a stale execution artifact (likely from a run where the slice was `[:5000]`). **Never trust a notebook's stored output**; re-run it. This is exactly the class of bug that makes research notebooks non-reproducible.
2. **The retrain section pads *before* slicing, the train section pads *after*.** `pad_sequences` on 25 000 rows then `[:1000]` (cells 17–18) vs `[:3000]` then `pad_sequences` (cells 5–7). Functionally equivalent here, but it means the two sections are not doing the same thing, which is how subtle train/eval skew gets introduced.
3. **`load_model("lstm_imdb_model.h5")` requires the file to be in the CWD.** In Colab, a runtime restart wipes it. The video never mentions this, and it is the reason the load cell fails the first time on camera ("model loading is not defined" [06:34:47–35:00]).
4. **`model.save('.h5')` does not save the tokenizer or the vocabulary.** The decoded-review cell can only work because `imdb.get_word_index()` re-derives the mapping from the Keras dataset. Deploy that model anywhere else and you cannot decode its own outputs. **A model artifact without its preprocessing is not an artifact.**
5. **`padding='post'` + `truncating='post'` on IMDb specifically loses the review's verdict.** See §6.2 and the decoded text in §6.4.
6. **`Dense(1, sigmoid)` + `binary_crossentropy` is the correct pairing; `Dense(1)` + `binary_crossentropy` with `from_logits=True` is the numerically better one.** The sigmoid-then-BCE form can saturate: for a very wrong prediction the gradient is `σ'(z)·(...) ≈ 0`, so a *confidently wrong* example contributes almost nothing. `from_logits=True` computes the loss from `z` directly with a stable formulation. On a 3 000-row demo it will not matter; on a real training run it does.
7. **Keras 3 counts optimizer state in "Total params."** Covered in §6.3 — a 3× model-size overestimate if you read it naively.
8. **The decoder in the notebook uses `initial_state=[state_h, state_c]` and never receives the encoder's output sequence.** With `return_sequences=False` on the encoder (which is how the notebook reuses `layers[2]`), there *is* no output sequence to attend over. Adding attention requires rebuilding the encoder with `return_sequences=True` — you cannot bolt attention onto a checkpoint whose encoder discarded its outputs.
9. **Attention weights are not explanations.** The instructor's `it → animal` figure [07:30:25] is a *visualization*, and Jain & Wallace (2019) showed attention distributions can be substantially perturbed without changing predictions. Use attention maps for debugging, never for a compliance claim.

---

## 11. Cost, Compute & Memory

### 11.1 The notebook's measured costs (real numbers from the saved outputs)

| Run | Data | Batch | Epochs | Wall clock (Colab GPU) | Throughput | Result |
|---|---|---|---|---|---|---|
| Classification | 3 000 rows | 64 | 1 | **89 s** (71 steps, 1 s/step) | 34 rows/s | acc 0.5011 |
| Retrain | 1 000 rows | 64 | 1 | **22 s** (15 steps, 1 s/step) | 45 rows/s | acc 0.5386 |
| "Summarization" | 1 000 × 200 encoder + 1 000 × 50 decoder | 32 | 1 | **67 s** (29 steps, 2 s/step) | — | loss = ln(8000) |

**The 1 s/step figure is the whole story.** Forty-two optimizer steps in 89 seconds on a GPU. For contrast, a 7B transformer with LoRA at batch 4 × 512 tokens runs ~2 steps/s on the same T4 — and each of those steps sees 4 × 512 = 2 048 tokens against the LSTM's 64 × 200 = 12 800 token-positions but only 64 *sequences*. **Normalized per sequence, the LSTM is ~30× slower per gradient step and needs two to three orders of magnitude more steps.**

### 11.2 VRAM formula — LSTM vs transformer

**LSTM training VRAM** (per layer, per batch):

```
Params         :  P = 4·d_h·(d_x + d_h + 1)
Weights fp32   :  4P bytes
Gradients fp32 :  4P bytes
Adam m, v fp32 :  8P bytes
Activations    :  ~B · T · d_h · 4 bytes × (number of stored gates)
Total ≈ 20P + B·T·d_h·32 bytes
```

Worked for the notebook at `d_h=256, d_x=128, B=64, T=200`:
- `P = 394 240 + 1 280 000 + 257 = 1 674 497` (include the embedding)
- Weights + grads + Adam = `20 × 1.67e6 = 33.5 MB`
- Activations (cuDNN stores ~6 tensors of `B×T×d_h`): `64 × 200 × 256 × 4 × 6 = 78.6 MB`
- **Total ≈ 112 MB.** The notebook's 89 s on a T4 was not memory-bound. It was **latency-bound**, which is worse: you cannot fix a latency bound by buying a bigger GPU.

**Transformer training VRAM** (the standard mixed-precision formula):

```
Weights fp16     :  2P
Gradients fp16   :  2P
Adam m, v fp32   :  8P          (master weights 4P + m 4P + v 4P = 12P if using AMP)
Activations      :  ~ 2 · B · n · d_model · L · k   (k ≈ 10-20 with recompute off)
KV cache (decode):  2 · L · n_kv · d_head · bytes  per sequence
Total (AMP, no recompute) ≈ 16P + activations
```

Worked for **7B LoRA fine-tuning, `B=4, n=1024, L=32, d_model=4096`**:
- Base weights fp16: `2 × 6.74e9 = 13.5 GB` (frozen, no grads, no Adam)
- LoRA params (`q,k,v,o,gate,up,down` at rank 8): ~`7 × 4096 × 8 × 2 = 0.46 M` per layer × 32 = **14.7 M trainable** → grads 30 MB, Adam 120 MB
- Activations `2 × 4 × 1024 × 4096 × 32 × 15 ≈ 16 GB` — **this is the dominant term**
- **Total ≈ 30 GB** → fits a 40 GB A100 with gradient checkpointing off, or a 24 GB 4090 with checkpointing on (`~14 GB`)

**Cross-check the LSTM's place in this table:** a 1.67 M-parameter LSTM used 112 MB and 89 s. A 7B transformer with 14.7 M trainable parameters uses 30 GB and ~4 s/step for 4 096 tokens. The LSTM is 100× smaller and ~20× slower per unit of data. That ratio — **smaller and slower** — is the definition of a superseded architecture.

### 11.3 Worked cost example: "fine-tune on a domain corpus in the LSTM era vs now"

**Scenario:** 50 000 domain support tickets, 4-class routing.

| Approach | Steps to a usable model | GPU-hours | $ at \$2/GPU-h (A100 spot) | Notes |
|---|---|---|---|---|
| **LSTM from scratch (2016 recipe)** | ~30 epochs × 50 000/64 = 23 400 steps | 23 400 × 1 s / 3600 = **6.5 h** — *if* it converges | **$13** | But it will not reach the same accuracy, and you redo it for the next task |
| **LSTM with ULMFiT-style pretraining** | Pretrain on 1 B tokens first: ~2 000 GPU-h | **2 000 h** | **$4 000** | Then 2 h per task. This is why nobody did it |
| **BERT-base fine-tune (2019 recipe)** | 3 epochs × 50 000/32 = 4 700 steps @ ~4 steps/s | **0.3 h** | **$0.65** | Plus the one-time cost of BERT's pretraining (64 TPU-v3-days, ~$7 000 in 2019 dollars, amortized to zero) |
| **LoRA on Llama-3.1-8B (2026 recipe)** | 2 epochs × 50 000/16 = 6 250 steps @ ~1.5 steps/s | **1.2 h** | **$2.30** | Best accuracy; 0.5 % of parameters; runs on a 24 GB consumer card with QLoRA |

**The headline:** the *marginal* cost of a new task fell from a from-scratch training run to a 1–2 GPU-hour LoRA job — a ~100× reduction in engineering time and a ~1000× reduction once you stop re-pretraining the base. The instruction "just fine-tune it" only became cheap because the *base model* became reusable, and the base model became reusable because the architecture standardized. **Everything in this module is in service of explaining why "just fine-tune it" was not a sentence anyone could say before 2018.**

### 11.4 Inference cost — the crossover table

Per-token decode cost at batch 1, fp16, on an A100 (assume memory-bandwidth-bound at 1.55 TB/s):

| Model | Weights | KV/state per token | Bytes/token moved | Tokens/s (batch 1) |
|---|---|---|---|---|
| LSTM, `d_h=256`, 1 layer | 6.4 MB | 2 KB (constant) | 6.4 MB | ~240 000 |
| LSTM, `d_h=4096`, 32 layers (hypothetical) | 2.1 GB | 512 KB (**constant**) | 2.1 GB | ~740 |
| Llama-3-8B (32 L, GQA-8) | 16.1 GB | 128 KB (**grows with n**) | 16.1 GB | ~96 |
| Llama-3-70B (80 L, GQA-8) | 141 GB | 320 KB (grows) | 141 GB | ~11 |

**Read the last column together with the third.** At batch 1, decode is bandwidth-bound on *weights*, so a small model is fast regardless of architecture. The architecture difference appears when you either (a) batch heavily — then the weights amortize and the KV cache becomes the dominant traffic (a transformer's KV traffic grows with `n·B`, an LSTM's does not), or (b) run very long contexts. **This is the honest version of "RNNs are cheaper at inference": they are cheaper when `B·n` is large enough that KV-cache traffic dominates.**

---

## 12. Evaluation — How To Know It Worked

### 12.1 The metrics, and how each one lies

| Metric | What it measures | How it lies to you |
|---|---|---|
| Accuracy | Fraction correct | Degenerate on imbalanced data. 95 % accuracy on 5 % positives = a majority-class predictor |
| F1 (macro) | Per-class recall/precision balance | Hides which class is failing; always print per-class |
| AUC-ROC | Ranking quality | Insensitive to calibration. A model with AUC 0.95 can emit 0.9 for everything |
| Calibration (ECE / reliability diagram) | Does "0.9" mean 90 %? | Not reported by default; **the single most important metric if you gate on a threshold** |
| BLEU / ROUGE | n-gram overlap | Rewards generic output; a summarizer that copies the first sentence scores respectably |
| Perplexity | Model's own loss, exponentiated | Only comparable across models with the **same tokenizer**. Two models with different vocabs have incomparable perplexities |
| Task accuracy on a *held-out slice of the original distribution* | Forgetting | Omitted by default in fine-tuning work; this is how you catch catastrophic forgetting |

### 12.2 The LSTM-era protocol vs the correct protocol

The notebook's protocol: `validation_split=0.1` on a 3 000-row tensor, one epoch, no test set at all (the test split is discarded with `_`), no calibration, no per-class metrics, no length-bucketed analysis.

**The protocol that would have caught the notebook's bugs in five minutes:**

```python
import math, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve

# 1. DEGENERACY CHECK (do this FIRST). If loss ~= ln(num_classes) the model
#    learned nothing -- regardless of what the accuracy column says.
print("uniform floor (binary):", math.log(2))       # 0.6931
print("uniform floor (8000):  ", math.log(8000))    # 8.9872

probs = model.predict(X_val).ravel()
preds = (probs > 0.5).astype(int)

# 2. Per-class metrics. Never accuracy alone.
print(confusion_matrix(y_val, preds))
print(classification_report(y_val, preds, digits=4))
print("AUC:", roc_auc_score(y_val, probs))

# 3. Calibration -- required if you gate on a threshold.
frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
print("reliability:", list(zip(mean_pred.round(3), frac_pos.round(3))))

# 4. LENGTH-BUCKETED accuracy -- catches truncation and length non-generalization.
lengths = (X_val != 0).sum(axis=1)
for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 201)]:
    m = (lengths >= lo) & (lengths < hi)
    if m.sum() >= 20:
        print(f"len [{lo:3d},{hi:3d}) n={m.sum():5d} acc={(preds[m]==y_val[m]).mean():.4f}")

# 5. FORGETTING CHECK -- required whenever you fine-tune an existing checkpoint.
drift = np.abs(model.predict(X_base_val) - base_preds_before).mean()
print(f"mean |prediction drift| on base distribution: {drift:.4f}")
```

**What each block catches in the companion notebook:** block 1 catches *both* broken runs (`0.6942 ≈ ln 2 = 0.6931` for classification; `8.9872 = ln 8000` for summarization); block 3 catches the `0.4902 → "Negative"` coin-flip-as-label; block 4 catches the `max_len=200` + `truncating='post'` loss on long reviews; block 5 catches the "retrain" section's damage to the original classifier.

### 12.3 Evaluating an encoder–decoder specifically

Standard NMT/summarization evaluation, which the notebook never reaches:

1. **BLEU / ROUGE** — fast, cheap, weak. Report **SacreBLEU** (tokenizer-controlled, comparable across papers) not BLEU.
2. **Length-bucketed BLEU** — *the* diagnostic for the bottleneck. Plot BLEU against source length. A no-attention encoder–decoder's curve falls off a cliff past 30 tokens; an attention model's stays flat. If your curve falls, you have the bottleneck, not a data problem.
3. **Copy/coverage analysis** — fraction of target tokens that appear in the source. A summarizer with poor coverage is either hallucinating or dropping content.
4. **Free-running vs teacher-forced perplexity.** Compute both. The gap *is* your exposure bias. A gap > 2 nats is a scheduling problem; a gap of 0 is suspicious (you are probably leaking the target).
5. **Human or LLM-judge spot check** on 50 outputs, stratified by source length. Cheap, and it catches the failure modes that n-gram metrics are blind to.

---

## 13. Comparison Tables

### 13.1 The master table — five architectures, seven axes

Read this as the definitive reference for the module. "Path length" is the number of *nonlinear layers* a signal must traverse between two positions `k` apart.

| Axis | **Simple RNN** | **LSTM** | **GRU** | **Transformer** | **SSM / Mamba** |
|---|---|---|---|---|---|
| **Sequence mixing** | Recurrence `h_t = tanh(W h_{t−1} + U x_t)` | Gated recurrence with a separate additive cell state | Gated recurrence, single state | `softmax(QKᵀ/√d_k)V` | Linear recurrence `x_k = Āx_{k−1} + B̄u_k` |
| **Path length between 2 tokens** | `O(k)` — `k` nonlinear steps | `O(k)` for `h`; `O(k)` diagonal for `C` | `O(k)` | **`O(1)`** | `O(k)` but **linear** (no nonlinearity between steps) |
| **Parallel across time (training)** | No | No | No | **Yes — one GEMM over `n`** | Yes — parallel scan (Mamba-1) or matmul form (Mamba-2) |
| **Per-step gradient factor** | `‖diag(1−h²)·W_hhᵀ‖` = `λ` (unbounded) | `diag(f_t)` on the cell path (bounded in `(0,1)`); matrix products elsewhere | `diag(z_t)` added directly on the readable state | Residual identity `I` + attention (path 1) | `diag(Ā)` — bounded, and `A` is structured (HiPPO/diagonal) |
| **Long-range dependency** | Fails past **10–20** steps | Usable to **~100–200** steps (measured); 1000+ with orthogonal init + clipping | ≈ LSTM, slightly worse on very long | Unbounded in principle; limited in practice by head capacity and training data | Linear decay, no exponential blowup; **weak on exact recall** |
| **Inference state per token** | `d_h` floats | `2·d_h` floats | `d_h` floats | **KV cache: `2·L·n_kv·d_head` bytes, grows with `n`** | `d_state·d_model` per layer (Mamba-2: `d_state=128` typically), **constant** |
| **Training FLOPs per token per layer** | `≈ 2d_h² + 2d_h d_x ≈ 4d²` | `≈ 8d²` (4 gates) | `≈ 6d²` (3 gates) | `≈ 12d² + 4nd` (projections + FFN + attention) | `≈ 6d²` (in-projection, SSM, out-projection) |
| **Training wall-clock scaling in `n`** | **`O(n)` serial** — throughput ∝ `1/n` | `O(n)` serial | `O(n)` serial | **`O(1)` until `n > 3d_model`, then `O(n²)`** | `O(n)` but parallel — throughput flat in `n` |
| **Inference cost per token** | `O(d²)` weights | `O(d²)` weights | `O(d²)` | `O(d²)` weights **+ `O(n·d)` KV traffic** | `O(d²)` weights, **no growth in `n`** |
| **Params at `d=4096`, 32 L (recurrence only)** | 0.54 B | **2.15 B** | 1.61 B | attention 2.15 B + FFN 4.33 B = **6.48 B** | ≈ 1.61 B |
| **Length extrapolation** | No position scheme needed; degrades with `k` | Same | Same | Depends on PE: sinusoidal poor, RoPE moderate, ALiBi strong | Good; the decay is a learnable/structured function of `Δ` |
| **Causality** | Free | Free | Free | Needs an explicit mask | Free |
| **Representative models** | Elman RNN, 2015-era baselines | AWD-LSTM (ULMFiT), GNMT, DeepSpeech-2 | Cho et al. 2014, many edge models | GPT-4/5-class, Llama 3/4, Qwen 3, Mistral, DeepSeek | S4, Mamba, Mamba-2, Jamba (hybrid), Zamba, Bamba, RWKV, RetNet |
| **2026 verdict** | Obsolete | Obsolete for text; alive on edge/streaming | Obsolete for text; alive on edge/streaming | **The default** | Frontier for long-context hybrids; not a standalone replacement |

### 13.2 The instructor's own comparison table, completed

He presents an RNN / LSTM / Transformer table at [07:43:38–45:16]. Reproduced with the gaps filled:

| His row | His values | Verdict |
|---|---|---|
| Architecture | "recurrent / recurrent with a gate / attention-based, no recurrence" | **Correct** |
| Handle long-term dependency | "no / better than RNN / —" | **Correct**, but "better" needs the number: ~10–20 → ~100–200 steps |
| Training parallelism | "sequential / sequential / parallel — fast" | **Correct.** The word "fast" hides the 100× |
| Vanishing gradient | "common / less common, gates help but not at all / —" | **Correct and well-phrased.** "Gates help but not at all [eliminate it]" is the right nuance |
| Memory / context window | "small / medium, better than RNN / large" | **Correct directionally.** The precise statement is that RNN/LSTM memory is `O(1)` in `n` and the transformer's is `O(n)` — the transformer's "larger context" costs memory, it is not free |
| Suitability by length | "small text / medium sequence / longer text" | **Correct** |
| Positional tracking | "built-in / built-in / needed because of parallelism" | **Correct** |
| Attention support | "no / no / yes — plus cross-attention and masked attention" | **Correct** |
| Inference speed | "slow / slow / fast" | **Only partly right.** At batch 1 a small LSTM can be *faster* than a large transformer. The correct statement is *per unit of model capacity and at large batch/long context*, the transformer wins because it amortizes weights over `n` (see §11.4) |
| Pretrained availability | "not available / not available / available — BERT, GPT" | **Correct, and this is the row that matters for fine-tuning** |

### 13.3 Where each family still wins — the honest 2026 scorecard

| Task | Winner | Why |
|---|---|---|
| Instruction following, chat, reasoning | Transformer | Pretrained scale + tooling |
| Long-context document QA at 128K | Transformer (with FA + GQA) or hybrid | Attention's exact recall |
| 1M-token streaming with bounded RAM | **SSM / hybrid** | Constant state |
| Real-time speech recognition on-device | Transformer (streaming variants) or Conformer; LSTM in legacy stacks | Accuracy |
| Wake-word detection on a Cortex-M4 | **RNN/LSTM or DS-CNN** | 100 KB RAM budget; no cache |
| Sensor anomaly detection with 500 labeled windows | **LSTM/GRU** | The inductive bias beats a transformer at this data volume |
| Associative recall over a long input | **Transformer** | A fixed state provably cannot; see Jelassi et al. 2024 |
| Video generation at 100K+ tokens | Hybrid (attention + SSM) | Both properties needed |
| Model-based RL world models | **Recurrent (RSSM)** | Belief states are recurrent by construction |

---

## 14. Debugging Playbook

Ordered by frequency. "Diagnostic" is the single command or check that discriminates.

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | Loss ≈ `ln(num_classes)` and flat | Model learned nothing: underfit, dead gradient, or (seq2seq) noise targets | Compute `ln(C)` and compare to 4 decimals. Also print `len(x)/batch` = number of steps | If steps < 500, train longer. If targets are synthetic, fix the data |
| 2 | Train loss falls, val loss rises | Overfitting | Plot both; look for the divergence epoch | Dropout 0.1–0.5, reduce `latent_dim`, early stopping, more data |
| 3 | Train and val loss both flat, both ≈ floor | Learning rate too low, or the gradient is dead | Log `grad_norm` per step. If it is `< 1e-6`, the gradient is dead | Raise LR; check init; check for `tanh` saturation (`|h| → 1` fraction) |
| 4 | Loss spikes then NaN at step ~100–500 | Exploding gradient | Log pre-clip `grad_norm` | `clipnorm=1.0`; halve the LR; check for a bad batch (all-pad sequence) |
| 5 | Loss goes to exactly 0.0 | Label leakage: the target is in the input, or the eval set overlaps the train set | Shuffle labels at random and retrain. If loss stays ~0, you have leakage | Fix the split; for seq2seq, verify the decoder input is *shifted* |
| 6 | Loss decreases; generations are incoherent | Exposure bias (teacher forcing at train, free-running at inference) | Compute free-running vs teacher-forced perplexity; the gap is the bias | Scheduled sampling; add noise to decoder inputs; shorten the horizon |
| 7 | Accuracy 0.99 but the model is useless | Class imbalance + majority-class collapse | `confusion_matrix`; per-class recall | Class weights, resampling, focal loss; report macro-F1 |
| 8 | `val_accuracy` bounces ±0.05 between epochs | Validation set too small | `n_val`; standard error is `√(p(1−p)/n_val)` | Use ≥ 1 000 validation rows or cross-validate |
| 9 | Training 5–10× slower than expected | `recurrent_dropout > 0` (cuDNN disabled), or `unroll=False` on a short static `T`, or no `cuDNN` (running on CPU) | `tf.config.list_physical_devices('GPU')`; set `recurrent_dropout=0` and re-time | Set `recurrent_dropout=0`; use `dropout` instead |
| 10 | GPU utilisation at 5–15 % with high memory use | Latency-bound: too many small kernels (long `T`, small batch) | `nvidia-smi dmon`; or profile with `torch.profiler` / TF Profiler | Increase batch size; reduce `T` and use TBPTT; this is the architectural limit — see §4.4 |
| 11 | OOM on a batch size that "should" fit | Activations from full BPTT over long `T`, or an attention matrix materialized | Compute `B·T·d_h·4·6`; for attention, `B·h·n²·2` | TBPTT, gradient checkpointing, smaller `T`; for attention, FlashAttention |
| 12 | `NaN` in the loss but finite gradients | `log(0)` in a hand-rolled loss, or a `softmax` over an all-`−inf` row (a fully-masked position) | Check for all-pad rows in the batch | Mask the loss; `-1e9` not `-inf` for masks; `from_logits=True` |
| 13 | Model works on short inputs, fails on long ones | Trained with `max_len` truncation; TBPTT cap; or absent positional generalization | Length-bucketed accuracy (§12.2 block 4) | Train at the deployment length; head+tail truncation; RoPE scaling |
| 14 | Decoder produces the same token forever | The `<eos>` token is never in the training targets, or the loss ignores the first position | Inspect the target tensor: does it contain `<eos>`? | Add `<eos>` to targets; check `dec_target = target[:, 1:]` alignment |
| 15 | "Fine-tuned" model worse on the original task | Catastrophic forgetting (full-weight updates, small new dataset) | The drift check from §12.2 block 5 | LoRA/QLoRA instead of full FT; mix 5–20 % replay data; lower LR; fewer epochs |
| 16 | Identical loss to 6 decimals across different runs and different seeds | The model is not reading its input at all (e.g. it is only reading the mask, or the input is all zeros) | Permute the input; the loss must change | Check the data pipeline end to end. Assert `x.std() > 0` |
| 17 | `AttributeError: 'Model' object has no attribute 'load_model'` | `load_model` is a module-level function, not a method | — | `from tensorflow.keras.models import load_model` (the exact mistake in the video at [06:34:47–35:06]) |
| 18 | `NameError: name 'imdb' is not defined` | `imdb` was never imported; only `tf.keras.datasets.imdb` was used | — | `from tensorflow.keras.datasets import imdb` (the exact mistake at [06:31:44–32:21]) |
| 19 | Val metric is 0.0000 on a seq2seq task | The metric is exact-match accuracy over 50 timesteps at 8000 classes — `(1/8000)^50` | Compare to `1/V` per position | Use per-token accuracy or perplexity, not sequence accuracy |
| 20 | Loss is `ln(V)` and stays there, loss curve perfectly flat | Targets are uniform random (or all zeros, or all the same token) | `np.unique(y).shape`; `y.std()` | Fix the data. This is §6.5 verbatim |

---

## 15. Applied Case Studies

### 15.1 "Our LSTM classifier was fine for two years and then the domain shifted"

**Situation.** A 200-person SaaS company in 2019 shipped an LSTM ticket router: 4 classes, word-level vocab of 30 000 built on their own 180 000 tickets, 2-layer LSTM at `d_h=512`, trained 8 epochs on 2×V100. It reached 91 % accuracy and ran at 4 ms/ticket on CPU.

**Why the technique was chosen then.** It worked, it was cheap at inference, and no pretrained alternative existed for their domain vocabulary.

**What broke.** In 2023 they added a product line whose vocabulary (product SKUs, acronyms) was not in the 30 000-word vocabulary. Every new-product term hit `<UNK>`. Accuracy on the new-product tickets was 62 %; overall accuracy fell to 79 %.

**Why they could not just fine-tune it.** The vocabulary was baked into the embedding table. Adding SKUs required resizing the embedding, which required reinitializing those rows, which meant training them from scratch on a corpus that did not contain them at scale. There was no checkpoint to return to and no tokenizer to swap.

**The migration, with numbers.** They moved to `distilbert-base-uncased` + a 4-class head in 2023. Note that a subword tokenizer does not have an OOV problem — any string is covered.
- Data: the same 180 000 tickets, 3 epochs, batch 32, `lr=2e-5`
- Hardware: 1×V100, **4.5 h**, ~$6 at spot pricing
- Result: **94.1 %** overall, **89 %** on the new product line
- Inference: `distilbert` at 66 M params quantized to INT8 → **11 ms/ticket on 4 CPU cores** (vs 4 ms for the LSTM). They accepted the 2.7× latency for the accuracy.
- **The lesson:** the embedding-vocabulary lock-in is not a hyperparameter. It is the architectural property that made transfer impossible, and it is why "just add the new words and retrain the embedding" is not a thing you can do to a 2016 model.

### 15.2 The encoder–decoder bottleneck, measured

**Situation.** A medical-abstract summarizer, 2021, built on a seq2seq LSTM because the compliance team wanted an on-prem, auditable model. 40 000 paper-abstract pairs, encoder `d_h=512` 2-layer bidirectional, decoder `d_h=1024` 2-layer, trained 20 epochs on 4×V100 (31 h).

**Symptoms.** ROUGE-1 of 0.31 overall looked acceptable. But reviewers noticed the summaries were fluent and *wrong* — they invented findings.

**The diagnosis.** Length-bucketed ROUGE (the §12.3 step 2 plot):

| Source length (tokens) | ROUGE-1 | Coverage (target tokens in source) |
|---|---|---|
| 0–150 | 0.37 | 0.81 |
| 150–300 | 0.33 | 0.72 |
| 300–450 | 0.26 | 0.54 |
| 450+ | **0.14** | **0.29** |

The curve collapses past 300 tokens and coverage collapses with it. **That is the bottleneck, not a data problem.** The encoder's `(h, C)` is 2 048 floats; a 450-token abstract at `d_x=300` is 135 000 floats of embedding content. The model cannot hold it and starts generating plausible text instead.

**First fix that did not work:** 2× the hidden size (`d_h=1024` encoder / 2048 decoder). ROUGE-1 on 450+ went from 0.14 to 0.19. Cost: 4× parameters, 3.1× training time. **Doubling a bottleneck's width does not remove the bottleneck** — the compression ratio improved by 2× while the content grew by 3× more than that.

**Fix that worked:** Bahdanau attention over the encoder outputs. `d_h=512` unchanged, +2.1 M parameters (+3 %), training time +18 %. Result: **450+ ROUGE-1 = 0.33**, overall 0.31 → **0.41**, coverage 0.29 → 0.74. The model now had a *direct, content-addressed path* to every source position instead of one fixed-size vector.

**What the case teaches:** this is the exact empirical result of Bahdanau et al. (2014), reproduced. Attention was not a generational leap — it was a 3 % parameter increase that fixed an information-theoretic problem, and it arrived three years before the transformer.

### 15.3 Where the LSTM still wins: a 30 KB wake-word model

**Situation.** 2025. Deploy "Hey <product>" on a Cortex-M33 at 64 MHz with 256 KB of RAM and no network. Budget: 20 KB of model weights, 30 ms of latency, < 100 µA average.

**Why the transformer loses here, with arithmetic.** A 4-layer transformer at `d_model=64`, `n=32` frames: KV cache = `2 × 4 × 32 × 64 × 2 B = 32 KB` *and it grows to* `2 × 4 × n × 64 × 2 B` for any `n`. At `n = 200` frames (2 s of audio at 100 fps) that is 200 KB — the entire RAM budget. Plus attention's softmax needs `exp()` over `n` values, and a Cortex-M33 has no fast transcendentals.

**The LSTM's numbers.** A 2-layer GRU at `d_h=64`, input 40-dim mel frames:
- Parameters: `3 × (40×64 + 64² + 2×64) × 2 layers = 3 × 6816 × 2 = 40 896` — at INT8 that is **41 KB**. A `d_h=48` variant fits in 24 KB.
- Inference state: `2 layers × 48 floats = 384 bytes`, **constant regardless of how long the microphone streams**.
- Compute: `6 × 48² = 13 824` MACs per frame. At 64 MHz with CMSIS-NN SIMD, that is **~0.4 ms/frame**, comfortably inside a 10 ms frame budget.
- No KV cache, no softmax, no `n²` term, no positional encoding.

**Result.** 96.1 % accuracy on a 12-word custom keyword set, 22 KB INT8, 0.4 ms/frame, 8 µA in always-on mode. A distilled transformer of equal accuracy was 3.1× larger and could not run in the RAM budget at the required stream length.

**The generalization to carry:** *the transformer's advantages are all functions of scale — pretraining corpus, parameter count, attention width. Below a certain scale, the inductive bias of a recurrence is worth more than parallelism you cannot use.* This is the counterweight to §4.4, and it is a real engineering position, not nostalgia.

### 15.4 The migration playbook: LSTM → transformer without a flag day

Shadow-deploy the transformer on 100 % of traffic (log both, serve the LSTM) → gate on "transformer accuracy ≥ LSTM − 0.5 pp on the *live* distribution" → build the eval harness and confirm it alarms on a deliberately injected tokenizer bug → 5 % → 50 % → 100 %, keeping the LSTM warm on 1 % shadow traffic for 30 days. Gate each stage on per-class recall and p99 latency, not on aggregate accuracy.

**What went wrong first:** phase 2's p99 was 31 ms instead of 25 ms because the INT8 quantized model silently fell back to fp32 for the attention softmax on some inputs. Fixing the `QConfigMapping` to use a supported INT8 softmax brought p99 to 19 ms. **What they would change:** they decommissioned the LSTM's *vocabulary file* along with the model, then discovered compliance needed to reproduce historical decisions for 7 years. **A model you have deleted is a model you cannot audit** — the concrete reason legacy RNN artifacts outlive their serving path in regulated environments.

---

## 16. Production Considerations

### 16.1 The five things a pre-transformer model deployment could not do

| Capability | 2016 LSTM deployment | 2026 transformer deployment |
|---|---|---|
| Versioning | `lstm_imdb_model.h5` vs `..._updated.h5` in a Colab filesystem | `safetensors` + `config.json` in a registry, content-addressed, model card + git SHA |
| Rollback | Re-run the notebook; hope the vocab matches | Swap the adapter directory or the checkpoint revision — instant |
| Monitoring | None | Token drift, output-distribution drift, calibration tracking, refusal rate |
| Regression tests | None | 200 golden inputs with expected outputs, run on every checkpoint |
| A/B | Impossible — one model, one task, one architecture | Two adapters behind one base model; cost delta ≈ 0 |

**The root cause of the 2016 column is one sentence: without a shared base model there is nothing to version separately from the training run.** You cannot roll back a fine-tune when the fine-tune *is* the model. The reusable base is what makes MLOps possible at all, which is why CS-01's lifecycle framing (pretrain → adapt → serve) is downstream of this module's architecture argument.

### 16.2 `Beyond the video:` FlashAttention and what it did to the `O(n²)` argument

The instructor never mentions FlashAttention, and the "attention is `O(n²)`" argument as usually stated is now **half wrong in a way that matters for capacity planning.**

**What FlashAttention (Dao et al., 2022) does.** The naive attention implementation materializes the `n × n` score matrix in HBM, writes it, reads it for softmax, writes the softmax, reads it for the `AV` product. FlashAttention **tiles** Q, K, V into blocks that fit in SRAM (A100: 192 KB per SM) and computes the softmax **incrementally** using the online-softmax rescaling trick:

```
For each tile of K/V:
    S_tile  = Q_tile · K_tileᵀ / √d_k
    m_new   = max(m_old, rowmax(S_tile))
    P_tile  = exp(S_tile − m_new)
    ℓ_new   = exp(m_old − m_new)·ℓ_old + rowsum(P_tile)
    O_tile  = diag(exp(m_old − m_new))·O_tile + P_tile · V_tile
```

Note that `P_tile` is computed and *immediately consumed* — it never leaves SRAM. The `n × n` matrix is never written to HBM. **It is an exact computation, not an approximation** (unlike Linformer/Performer/Reformer, which change the math).

**What it changes:**

| Quantity | Naive attention | FlashAttention |
|---|---|---|
| FLOPs | `4n²d` | **`4n²d` — unchanged** |
| HBM traffic | `O(n² + n·d)` | **`O(n²d²/M)` where `M` = SRAM size** |
| Peak memory | `O(n²)` | **`O(n)`** |
| Wall clock (GPT-2, A100) | 1.0× | **2–4× faster**; up to 3× on long sequences |

**The numbers for a realistic case.** `n = 8192, h = 32, d_head = 128, fp16`:
- Naive: score matrix `8192² × 32 × 2 B = 4.29 GB` live, plus ~3 intermediates → `~13–17 GB` of transient memory, and `~4 × 4.29 GB = 17 GB` of HBM read/write traffic per layer per pass. At 1.55 TB/s that is **11 ms of pure memory traffic**. Useful math is `4n²d_model = 4 × 8192² × 4096 = 1.1 TFLOP` → 3.5 ms at 312 TFLOPS.
- **So naive attention spends 11 ms moving bytes to do 3.5 ms of math — it is memory-bound by 3×, which is exactly the opposite of what "O(n²) FLOPs" implies.**
- FlashAttention: traffic drops to `O(n²d²/M) ≈ 0.4 GB` → **0.26 ms** of traffic. Now the 3.5 ms of math dominates. Wall clock: **~4 ms instead of ~14.5 ms, a 3.6× speedup, with `O(n)` memory instead of `O(n²)`.**

**The correction to the standard interview answer.** "Attention is `O(n²)`" is true about **FLOPs** and false about **what actually binds**. Before FlashAttention, long-context attention was **memory-bandwidth-bound**, not compute-bound; after it, it is compute-bound. Practical consequences:
1. **`torch.nn.functional.scaled_dot_product_attention` (PyTorch ≥ 2.0) picks the FlashAttention kernel automatically** for fp16/bf16 with `d_head ∈ {16,32,64,128,256}`. You often get it for free.
2. In HF: `model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")` — requires `pip install flash-attn`.
3. **FlashAttention-2** (2023) improved work partitioning → ~2× FA1, hitting **50–73 % of A100 peak**. **FlashAttention-3** (2024) adds Hopper warp-specialisation and FP8 → **~75 % of peak** on H100.
4. **It does not make `n` free.** At `n = 128K` the FLOPs are `4 × 1.6e10 × 4096 = 2.8e14` per layer per sequence — 0.9 s of pure math at 312 TFLOPS. Long context is a *cost*, just a 4–10× smaller one than it was in 2021.

### 16.3 `Beyond the video:` Mamba / SSM — the modern challenger, and why the frontier is hybrid

**The mechanism.** A state-space model is a linear time-invariant system:

```
Continuous:   x'(t) = A x(t) + B u(t)
              y(t)  = C x(t) + D u(t)

Discretized (zero-order hold, step Δ):
              x_k = Ā x_{k-1} + B̄ u_k,   Ā = exp(ΔA),  B̄ = (ΔA)^{-1}(exp(ΔA) − I)·ΔB
              y_k = C x_k
```

Two properties follow immediately:

1. **Training is parallel.** This is a *linear* recurrence, so it unrolls to a convolution:
   `y = x ⊛ K` with `K = (CB̄, CĀB̄, CĀ²B̄, …)`. Convolutions are `O(n log n)` via FFT and fully parallel. **This is the single biggest structural advantage over the RNN: the recurrence is linear, so it can be parallelized in time.**
2. **Inference is `O(1)` per token.** Just carry `x_k`. No KV cache, no growth in `n`.

**What S4 → Mamba changed.** S4's `A, B, C, Δ` are **input-independent** — the same linear filter is applied to every token. That makes it a very good *smoother* and a very poor *router*: it cannot look at the input and decide what to remember. **Mamba's contribution (S6) is making `B, C, Δ` functions of the input:**

```
Δ_k = softplus(Linear(u_k))      ← the discrete step size, now input-dependent
B_k = Linear_B(u_k)
C_k = Linear_C(u_k)
```

`Δ_k` acts as a *gating* mechanism in continuous time: a large `Δ` means "reset and read the current token", a small `Δ` means "keep the state". That is functionally analogous to an LSTM forget gate — but now with a *linear* recurrence, so the parallel scan still applies. The scan is implemented with a hardware-aware kernel that keeps the state in SRAM, giving the reported **~5× higher generation throughput** than a same-size transformer.

**Mamba-2 / SSD (2024).** Restricts `A` to a scalar-times-identity, which enables a matmul-based formulation. The *Structured State Space Duality* result shows this restricted SSM is equivalent to **masked attention with a semiseparable (low-rank-structured) mask**. Practically: 2–8× faster than Mamba-1, and it makes the SSM-vs-attention relationship a continuum rather than a dichotomy.

**The honest limitations.**

| Limitation | Evidence | Consequence |
|---|---|---|
| Cannot do unbounded exact recall | Jelassi et al. (2024), *Repeat After Me: Transformers are Better than State Space Models at Copying* — a formal separation theorem | Fails on tasks requiring copy-a-token-from-position-k; RNNs/SSMs provably lose here |
| Weak at in-context learning / induction | Multiple 2024 ablations show pure Mamba underperforming on few-shot and retrieval tasks | Bad choice for a general assistant |
| Pure SSMs underperform at frontier scale | Every 2025–2026 frontier release is hybrid or pure attention | Not a drop-in replacement |
| Inference state must be *large* to compete | Mamba-2 uses `d_state = 128` or more; a small state is a tight bottleneck | The "constant memory" advantage is constant, but not free |

**The 2026 answer: hybrid.** The observed pattern is that a small number of attention layers restore the recall capability that linear layers lack, at a small fraction of the cost:

| Model | Architecture | Attention : linear ratio | Context |
|---|---|---|---|
| **Jamba** (AI21, 2024) | Mamba + attention + MoE, 52 B total / 12 B active | 1 : 7 | 256 K, fits 140 K in one 80 GB GPU |
| **Zamba** (Zyphra) | Mamba backbone + a *shared* global attention block | 1 block total | Parameter-efficient hybrid |
| **Bamba** (IBM) | Mamba-2 + attention | Alternating | 90 B-class |
| **Samba** (Microsoft) | Mamba + sliding-window attention | Interleaved | Long-context |
| **Hymba** (NVIDIA) | Attention heads + SSM heads in *parallel* per layer | Parallel, not sequential | Reaches better accuracy-per-parameter than either alone |
| **Qwen3-Next** (2025) | Gated DeltaNet (linear attention) + full attention | 3 : 1 | Frontier-scale production hybrid |
| **Nemotron-H** (NVIDIA) | Mamba-2 + attention + FFN | Mostly Mamba | 8B/56B |

**What to say in an interview:** *"Mamba is not an upgraded transformer — it is the strongest member of a different family, the state-space models. Its advantages are real: linear scaling and a constant-size inference state. Its limitation is equally real: a fixed-size state cannot do unbounded associative recall, which is a proven separation, not a tuning issue. That is why the 2025–2026 frontier settled on hybrids with a small number of attention layers, not on pure SSMs. If you are building for long-context streaming on a memory budget, benchmark a hybrid; if you are building a general-purpose assistant, start with attention."*

### 16.4 `Beyond the video:` "Attention Is All You Need" — the ablations that actually support the claim

The paper's argument is not "attention is better" in the abstract; it is a specific set of ablations. Reproduced from Table 3 (variations on the base model), WMT'14:

| Configuration | EN-DE BLEU | EN-FR BLEU | What it proves |
|---|---|---|---|
| **base** (`d_model=512, h=8, d_k=d_v=64, d_ff=2048, dropout=0.1`) | **27.3** | **38.1** | Reference |
| (A) **single attention head**, `d_k=d_v=64` | 25.8 | 37.4 | Multi-head is worth **+1.5 BLEU** on EN-DE. The smallest ablation effect of the set, and the most quoted |
| (B) `d_k = d_v = 16` (too small) | 25.1 | 36.5 | Head *width* matters more than head *count*. Reducing `d_k` hurts more than reducing `h` |
| (C) larger model (`d_model=1024, d_ff=4096, h=16`) | 26.5 | 38.4 | **The larger model is WORSE on EN-DE than base** — overfitting on a 4.5 M-sentence pair dataset. Bigger is not monotonically better |
| (D) no dropout | 25.9 | 37.6 | Dropout is worth **+1.4 BLEU**. The paper's most under-reported contribution |
| (E) **learned positional embeddings** instead of sinusoids | 27.1 | 37.9 | **Statistically identical to base.** The famous sinusoidal encoding is *not* load-bearing |
| **big** (`d_model=1024, h=16, d_ff=4096, dropout=0.3`) | **28.4** | **41.8** | The headline number, and it needed dropout 0.3 to get there |

*(Values as reported in the paper's Table 3 variant rows. Re-check the camera-ready before quoting in an interview.)*

**Three takeaways that interviewers actually want:**

1. **The single-head ablation is small (+1.5 BLEU).** If you claim "multi-head is essential," the paper's own numbers say it is worth ~5 % relative. What is essential is *attention itself*, and the paper's Table 2 shows that the *big* model with a *single* head matches the base model with eight (25.0 vs 27.3 — worse, but far from catastrophic).
2. **Row (E) is the most interesting line in the table.** Learned absolute position embeddings perform identically to sinusoids on in-distribution length. **The choice of positional encoding is not what made the transformer work** — removing recurrence is. Sinusoids were a bet on extrapolation that did not pay off, and it took until RoPE (2021) to get a positional scheme that actually generalizes.
3. **Row (C) vs `big` is the real lesson.** Same-ish architecture, different regularization, ±2 BLEU. **Training configuration matters more than architecture at this scale** — which is precisely the lesson that the LSTM era could never learn, because it had no scale at which to learn it.

Also from the same paper, the *training cost* of the claim: the base model trained in **12 hours on 8 × P100**; the big model in **3.5 days** on the same hardware. That is the number to cite when someone asks why the transformer won — it trained on 100× more data than any LSTM system could in the same wall clock.

### 16.5 `Beyond the video:` the KV-cache arithmetic, and what it means for serving

Covered numerically in §4.6.3. The serving consequences:

| Decision | Arithmetic | Consequence |
|---|---|---|
| Max batch size at 128K context, 80 GB A100, fp16 | Cache alone is 16.8 GB/session; weights 16.1 GB → `(80 − 16.1 − 4)/16.8 ≈ 3` sessions | Long context destroys throughput. You serve ~3 users per GPU, not 50 |
| The same at 8K context | Cache is 1.0 GB → `~59` sessions | The cache is why *context length*, not parameter count, sets your serving economics |
| Switch KV cache to FP8 | 128 KB/token → 64 KB/token | Doubles concurrency; costs ~0.1–0.3 quality points on most benchmarks |
| Switch MHA → GQA with 8 KV heads | 512 KB/token → 128 KB/token, **4×** | This one change is why Llama-3 can serve 128K at all |
| Use MQA (`n_kv=1`) | 512 KB → 16 KB/token, **32×** | Quality cost is real (1–2 points on some evals); used where memory dominates |
| Use prefix caching / PagedAttention | Reuse the cache for a shared system prompt | A 2 000-token system prompt cached once saves 2000 × 128 KB = 256 MB *per session* |
| Compute the LSTM alternative | 512 KB **constant**, for an equally-sized model | The only architecture with no `n` term. §15.3 |

**The number to memorize:** *KV cache ≈ `2 × n_layers × n_kv_heads × d_head × bytes` per token.* For Llama-3-8B fp16 that is **128 KB/token**, so 8 192 tokens = **1 GB**. At FP8, 512 MB. That single formula answers half the serving questions in a system-design interview.

---

## 17. Common Misconceptions

1. **"LSTMs solved the vanishing gradient problem."** They replaced `Π diag(tanh')·W_hhᵀ` with `Π diag(f_t)` on **one** path, leaving the `h`-paths unchanged. `Π f_t` still decays — with a learnable base in `(0,1)` instead of a weight-matrix spectrum. Gating buys ~one order of magnitude in usable dependency length (~10–20 → ~100–200 steps). §4.3.2.

2. **"The transformer has no vanishing gradient because attention has path length 1."** Half right. The gradient still traverses `L` blocks — **the residual's identity path, not attention, is what makes that traversal lossless.** Post-LN needs 4 000 warmup steps; Pre-LN does not. §4.6.1.

3. **"Adam fixes vanishing gradients because it normalizes by gradient magnitude."** Adam is scale-invariant, so it recovers a *uniformly* small gradient. It cannot recover a term that is `10⁻¹⁵` relative to another term in the same parameter's sum. §4.2.3. The single most common wrong answer in this topic.

4. **"RNNs are more parameter-efficient than transformers."** At equal `d`, an LSTM is `8d²` FLOPs/token/layer vs `12d² + 4nd` — ~1.5× cheaper **per token**, ~100× more expensive **per second** because it cannot use the tensor cores. Always state which. §4.4.2.

5. **"The `1/√d_k` prevents overflow."** It keeps the *variance* of the pre-softmax logits at 1. `Var(q·k) = d_k`; unscaled at `d_k = 128` the logit std is 11.3, the softmax saturates, and `∂softmax/∂z ≈ 2e-5`. Overflow is handled by the `−max` subtraction inside the kernel. §4.5.2.

6. **"The transformer is `O(n²)` and that is its fundamental limit."** Since FlashAttention the *memory* is `O(n)` and attention is no longer memory-bound; wall clock improved 2–4×. FLOPs are unchanged at `4n²d`. "`O(n²)`" is a FLOPs statement, not a capacity-planning statement. §16.2.

7. **"Multi-head lets each head look at a different part of the sequence."** Imprecise. Every head is a *dense* mixture over all positions; what differs is the subspace and the mixture pattern. The real structural difference is that each head's weights independently sum to 1. §4.5.3.

8. **"Positional encodings are essential to the transformer's success."** The paper's Table 3 row E shows learned position embeddings matching sinusoids exactly (27.1 vs 27.3 BLEU). What matters is that *some* position information exists. §16.4.

9. **"You can fine-tune an LSTM the way you fine-tune BERT."** No checkpoint to load, no shared tokenizer, no framework, and a vocabulary-locked embedding table. §5.2, §8.2. This is the module's thesis.

10. **"The encoder–decoder bottleneck is a vanishing-gradient problem."** It is an *information* problem: even at `λ = 1` with a perfect gradient, a 50-token source at `d_x=512` cannot fit into 1 024 floats of `(h, C)`. Attention fixed it by removing the compression step, not the gradient decay. §4.3.4(b).

11. **"Mamba is an improved transformer."** Different family — S4 → S5 → H3 → Mamba/S6 → Mamba-2/SSD. No attention anywhere in the lineage. §10.5, §16.3.

12. **"RNNs are dead."** Obsolete *for text at scale* — a different claim. Alive in wake-word detection, MCU inference, streaming sensor processing, RL world models, and as the basis of every SSM. §13.3, §15.3.

13. **"A lower loss means a better model."** The notebook's summarizer reports 8.9872 — exactly `ln(8000)`, the uniform floor, achieved by outputting the same distribution for every input. Compare every loss to the uniform floor before interpreting it. §6.5.

14. **"Model size = the 'Total params' line in `summary()`."** Keras 3 folds optimizer state into it. The notebook's model is 1.67 M parameters (6.39 MB), not 5.02 M (19.16 MB). §6.3.

15. **"The hidden layer count is what makes a network deep."** An RNN has two depth axes: stacked layers *and* time steps. With `T = 200` and one layer, BPTT already differentiates through 200 nonlinear steps — deeper than any practical feedforward net, which is why a 1-layer RNN has a 200-layer MLP's gradient problems. §4.2.

---

## 18. Key Takeaways

1. **The BPTT gradient is a product, and products of numbers near 1 collapse.** `∂h_t/∂h_k = Π J_j` scales as `λ^{t−k}`. At `λ=0.9`, 50 steps → `5.2e-3`; at `λ=0.5` → `8.9e-16`. That one line is the whole vanishing/exploding story.

2. **Exploding and vanishing are the same mechanism.** `λ>1` explodes, `λ<1` vanishes, and a real RNN does both along different directions because the Jacobian's singular values spread. Hence clipping *and* gating.

3. **The LSTM cell state is a gradient highway because `∂C_t/∂C_{t−1} = diag(f_t)` — diagonal, bounded, no weight matrix.** The word "diagonal" is the entire invention.

4. **Gating buys a constant factor, not a change of regime.** `Π f_t` is still exponential; it moves usable dependency from ~10–20 to ~100–200 steps and that is all.

5. **Sequential computation is an independent, equally fatal problem.** A matvec has arithmetic intensity 1 FLOP/byte; an A100 needs ~150 to saturate. **The RNN runs at ~0.6 % of peak; a GEMM-shaped transformer at 50 %.** Same FLOPs, ~100× the wall clock.

6. **The encoder–decoder bottleneck is an information problem.** A 50-token source at `d_x=512` is 25 600 floats compressed into 1 024. No gate fixes that; only a direct path does.

7. **Attention's `1/√d_k` is a variance argument:** `Var(q·k) = d_k`; at `d_k=128` the logit std of 11.3 saturates the softmax and kills its gradient.

8. **Below `n ≈ 3·d_model` the FFN costs more than attention.** At `d_model=4096` that crossover is 12 288 tokens — so "attention is `O(n²)`" is not the binding cost in the 4K-context regime.

9. **Attention was `O(n²)` in FLOPs and in memory until FlashAttention made the memory `O(n)` with a 2–4× wall-clock gain.** The FLOPs did not change; the binding constraint moved from bandwidth to compute.

10. **FFN ≈ 2/3 of a modern LLM's parameters, attention ≈ 1/3, embeddings ≈ 4 %.** Llama-2-7B's `6,738,415,616` is 64.2 % FFN. LoRA on `q_proj`+`v_proj` touches 0.50 % — which is why the modern recipe targets all seven projections.

11. **Pre-LN is why deep transformers train without warmup:** `∂(x+F(x))/∂x = I + ∂F/∂x`, and the `I` carries the gradient to layer 1 at full magnitude.

12. **KV cache = `2 × n_layers × n_kv_heads × d_head × bytes` per token.** Llama-3-8B fp16 = **128 KB/token**, so 8 192 tokens = 1 GB. An LSTM's state is constant in `n` — a permanent 1000×+ advantage at long context, bought with a compression bottleneck.

13. **Fine-tuning was impossible in the RNN era for five architectural reasons** — task-type, architecture, vocabulary, OOV, and objective mismatch — plus no pretrained checkpoint, no tokenizer reuse, no framework support, and per-task retraining from scratch.

14. **ULMFiT (2018) proved pretrain→fine-tune works on an LSTM** (18–24 % error reduction; 100 examples matching 100× the data). It failed to become universal because the architecture could not be scaled. **The recipe was right; the recurrence was the problem.**

15. **The companion notebook proves the thesis accidentally, three times:** 50.11 % accuracy after 42 optimizer steps; a retrain that silently feeds the embedding a vocabulary 10× smaller; and a "summarization" loss of `8.9872 = ln(8000)` on random targets.

16. **Do not over-generalize.** Recurrence and SSMs still win on constant-memory inference, streaming audio, MCU deployment, small-data time series, and RL world models. The 2026 frontier is *hybrid* — Jamba, Zamba, Bamba, Samba, Hymba, Qwen3-Next.

---

## 19. Self-Check Questions

1. Write the BPTT gradient expression for `∂L/∂W_hh` in a simple RNN, and explain why the terms with large `t−k` are exponentially small.
2. State the exact per-step gradient factor for the LSTM cell path and derive why it contains no weight matrix. Then give two reasons this does not solve vanishing gradients.
3. Why is `1/√d_k` in scaled dot-product attention? Give the variance calculation.
4. The instructor says a residual connection *concatenates* the input with the sublayer output. What does it actually do, and what breaks if you concatenate?
5. You have a 20 000-token context and attention is `O(n²)`. Compute the score-matrix memory in fp16 for 32 heads, then explain what FlashAttention changes and what it does not.
6. Give the five independent reasons the notebook's LSTM classifier cannot be repurposed for summarization, with the specific line of code that demonstrates each.
7. Derive the loss of exactly `8.9872` in the notebook's summarization run. What does it tell you about the data?
8. Compute the LSTM's arithmetic intensity at one time step, and the crossover batch size for an A100 in fp16. Then explain the ~100× wall-clock difference in one sentence.
9. Why does RoPE generalize to unseen positions when sinusoidal absolute encoding does not? What property must the encoding have?
10. Name two tasks where you would deploy an LSTM in 2026 instead of a transformer, with the specific number that decides it.

<details>
<summary><strong>Answers</strong></summary>

**A1.** `∂L/∂W_hh = Σ_t Σ_{k≤t} [∂L_t/∂h_t · Π_{j=k+1}^{t} diag(1−h_j²)W_hhᵀ] h_{k−1}ᵀ`. Every term contains a product of `(t−k)` Jacobians, bounded by `(γρ)^{t−k}` with `γ = max‖diag(1−h²)‖ ≤ 1` and `ρ = σ_max(W_hh)`. Large-`t−k` terms are exponentially small, so the gradient is dominated by short-range terms even though the long-range terms are *present*. The optimizer therefore fits the short-range objective.

**A2.** `∂C_t/∂C_{t−1} = diag(f_t)` because `f_t, i_t, g_t` depend on `h_{t−1}` and `x_t`, not on `C_{t−1}`. No weight matrix, entries in `(0,1)`, so it cannot explode. It does not solve vanishing because (i) `f_t` is learned and must drop below 1 to forget — at `f=0.9`, `0.9^50 = 5.2e-3`; (ii) the paths that matter (`∂h_t/∂h_{t−1}`, which reach earlier layers and the embeddings) go through `tanh(C_t)`, the output gate, and `U_·` matrices with `σ'`/`tanh'` factors ≤ 0.25 — structurally identical to the RNN Jacobian.

**A3.** With `q_i, k_i` iid mean 0 variance 1: `E[q·k] = 0`, `Var(q·k) = Σ Var(q_i k_i) = d_k`, so `std(q·k) = √d_k` — 11.3 at `d_k = 128`. Unscaled logits at that scale make the softmax nearly one-hot, so `∂softmax/∂z = p(1−p)` collapses to `~2e-5` and the head stops learning. Dividing by `√d_k` restores `Var = 1`.

**A4.** It adds: `z = LN(x + MHA(x))`. Concatenation grows the width by `d_model` per block (33× after 32 layers), changes every parameter count, and replaces the identity gradient term `I` in `∂(x+F(x))/∂x = I + ∂F/∂x` with a block matrix `[I ; ∂F/∂x]` that has no identity path. Without that `I`, gradients in a deep stack multiply into nothing.

**A5.** `n=20 000, h=32`, fp16: `20000² × 32 × 2 B = 25.6 GB` for the score matrix alone, with 3–4 such intermediates live → `~75–100 GB`. FlashAttention tiles Q/K/V into SRAM and does online-softmax rescaling so the `n×n` matrix never reaches HBM: memory drops to `O(n)`, HBM traffic to `O(n²d²/M)`, wall clock 2–4× better. **FLOPs are unchanged** at `4n²d`, and it is exact, not an approximation.

**A6.** (1) *Task-type*: `Dense(1, sigmoid)`+`binary_crossentropy` vs `Dense(8000, softmax)`+`sparse_categorical_crossentropy` (cells 8, 26–27). (2) *Architecture*: `return_sequences=False` discards every `y_t`; a summarizer needs it on both sides (cell 8). (3) *Vocabulary*: `num_words` 10000 → 1000 while the embedding keeps 10000 rows (cells 3, 17). (4) *OOV*: no subword units, no shared vocab file; cell 14 hardcodes the +3 offset from `imdb.get_word_index()`. (5) *Objective*: BCE on a scalar vs sparse categorical CE on `(batch, 50, 8000)` (cells 11, 27).

**A7.** Uniform softmax over `V=8000` classes: `−Σ(1/V)ln(1/V) = ln(8000) = 2.0794415 + 6.9077553 = 8.9871968`. The notebook reports `loss: 8.9872` and `accuracy: 1.2756e-04 ≈ 1/8000`. **The targets were `np.random.randint` — uniform — so the model learned the target distribution exactly and nothing else.** No signal in the data.

**A8.** One time step is a matvec: `8d²` FLOPs against `8d²` bytes of fp16 weights → **AI = 1 FLOP/byte**. An A100 offers `312e12/2.039e12 = 153` FLOP/byte at peak, so you need `n ≳ 100–150` to be compute-bound. The transformer batches `n` tokens into a GEMM with `AI = n`, so at `n=4096` it is 27× past the crossover. Same FLOP order per token, ~100× the wall clock, because the RNN never leaves the memory-bound regime.

**A9.** RoPE rotates Q and K such that `⟨RoPE(q,m), RoPE(k,n)⟩` depends only on `m−n`. Because the score is a function of the *relative* offset, the model learns a function of `Δ` that extends to unseen `Δ` (with degradation, mitigated by NTK/YaRN scaling). Sinusoidal *absolute* encoding adds `PE(pos)` to the embedding, so a position far beyond `L_train` produces an input pattern the model has never seen — no relative structure to generalize from.

**A10.** (1) **Wake-word spotting on a Cortex-M33 with 256 KB RAM**: the KV cache for a 200-frame transformer stream is `2 × 4 × 200 × 64 × 2 B = 200 KB` — the whole budget. A 2-layer GRU at `d_h=48` needs **384 bytes of state** and 24 KB INT8 of weights, at 0.4 ms/frame. (2) **Streaming at 128K+ tokens under a memory bound**: the LSTM's state is constant; Llama-3-8B's KV cache is 128 KB/token → 16.8 GB at 128K, more than the weights.

</details>

---

## 20. Cross-References

| Relationship | Module |
|---|---|
| Builds on | **CS-01** (pretraining/training lifecycle — the "why does a base model exist" framing), **CS-02** (transfer learning — the concept this module explains the pre-history of) |
| Needed by | **CS-06** (Hugging Face — the tooling that did not exist here), **CS-07** (fine-tuning BERT — the first architecture that made this module's problems go away), **CS-13 §6.8 + CS-11 §4.11** (LoRA/QLoRA — the parameter-efficiency argument that §4.6.2's 2/3-FFN split motivates; a dedicated "CS-23" module is planned but unwritten) |
| Contrasts with | **CS-04** (FT vs RAG vs agents — a different axis of "when to use which") |
| Reuses | **CH-05** (the equations reference), **IQ-05** (the interview bank) |

---

## Appendix A — Instructor's Verbatim Key Claims

| Timestamp | Quote | Status |
|---|---|---|
| 06:06:31–06:59 | "Recurrent means what? Recurrency. Recurrency means what? Again the same thing is happening… whatever output is being generating from this particular layer, again we are passing same output to the hidden layer itself and this is happening on a sequence basis. This is only is called the recurrency." | Correct |
| 06:07:31–07:43 | "Text is there and audio is there, DNA information is there, so those kind of data is called the sequence data where the sequence matters." | Correct |
| 06:13:09–13:45 | "The first thing is called the forget gate. Second is called the input gate and the third is called the output gate… the purpose of the forget gate is what needs to be forgot. The main purpose of input gate what needs to be add. And what was the purpose of the output gate? Like what needs to be passed as a output to the next time step." | Correct (semantics) |
| 06:15:35–15:54 | "In RNN there was a short-term memory, means we cannot remember like more than 10 to 15 words… but in LSTM basically we can recall that sentence up to 30 words." | **Imprecise** — see §10.1 |
| 06:16:25–16:39 | "Forget the long-term information, vanishing gradient issue was there and the slow training also was there and somewhat this problem was encountered with the LSTM also compared to the transformer." | Correct |
| 06:24:00–24:07 | "The encoder and decoder research paper was published in 2014 itself." | Correct (Sutskever/Cho) |
| 06:30:12–30:15 | "Here is the validation is split means 0.1 % data is going to be used for the validation." | **Wrong** — it is 10 %. See §10.8 |
| 06:35:44–36:08 | "LSTM model is not scalable means we are going to be retrained again on some new data set. So there is a chances it might forget the previous data set… first of all we cannot train it on very large sentences or we cannot even train it very huge amount of data set." | Correct — this is catastrophic forgetting + throughput, both right |
| 06:43:32–43:50 | "You have seen the task type mismatch. So first task basically it was a classification task and the next task basically was a summarization. So both are different. How it could be the possible? For this particular task the architectural also is a mismatch." | Correct |
| 06:45:53–46:02 | "The model basically which I shown you… I think we can only use the encoder weights but again it is not a prominent solution for the summarization task as well." | Correct and appropriately hedged |
| 06:46:55–47:26 | "ULM fit, that was a research which was introduced along with the transformer itself… inside the encoder decoder they have used LSTM model… the paper demonstrated a breakthrough… they first train a LSTM based encoder and decoder model on a large modeling… then they fine-tune this particular model on the text classification. The result was significantly good." | **Partly wrong** — ULMFiT is 2018 (after the transformer), is a single-stack AWD-LSTM (no encoder–decoder, no attention). The substance (pretrain then fine-tune) is right. See §10.6 |
| 06:47:39–47:55 | "LM based model can are computationally inefficient for the larger data. There is a lack of scalability due to the sequential processing. In transformer actually we have a parallel processing." | **The correct diagnosis.** This is the module's thesis |
| 06:48:47–49:14 | "See fundamental everything is a generation only… in classification also one word is being generated, summarization also some summarize sentence being generated, question answer normal answer is being generated, translation some translation part is being generated… fundamentally everything is a generated one but the variety of the task was different and one model, the LSTM model, was not able to handle this thing." | Correct, and the best insight in the two videos |
| 07:10:21–10:26 | "RNN forgets the long-term dependency, long-term information because of the vanishing gradient problem." | Correct |
| 07:11:12–11:39 | `new weight = old weight − dL/dw` | **Missing the learning rate** `η`. See §10.3 |
| 07:15:18–15:23 | "Cell state means long-term memory highway. It's going to be stored the long-term memory." | Correct |
| 07:17:50–18:03 | "This forget gate actually we are doing a cross the multiplication over here and because of that it is going to forget the information from this cell. And here in the input, we are doing addition." | **"Cross multiplication" is wrong** — it is element-wise. The multiply-vs-add structure is correct and is the key insight. See §10.4 |
| 07:22:43–22:57 | "This is called encoder decoder architecture. This was published in 2014 by Suska, a very prominent figure in the world of deep learning." | Correct (Sutskever; Cho et al. same year) |
| 07:24:10–24:18 | "This architecture was published in 2017 by the Google itself. This architecture actually it was a replacement on top of the RNN analysis." | Correct |
| 07:26:01–26:50 | "I am eating apple… while choosing apple. Both apple is trying to refer a different thing. First apple actually it is referring to the fruit and the second it is referring to the mobile phone. So using the RNN LSTM basically we were not able to capture this global context." | Correct as motivation for attention |
| 07:31:00–31:49 | "Residual connection means whatever input we are getting from here before the multi attention, as it is we are passing it over here and whatever thing we are going to be processed through the multi attention, both thing we are going to be concatenating." | **"Concatenating" is wrong** — it is addition. The stated purposes (regulate gradient flow, stabilize, speed up) are correct. See §10.2 |
| 07:32:22–32:48 | "NX means we can have as many as block… in the original research paper we were having six N block. All the encoder block is going to be identical." | Correct (N=6) |
| 07:28:39–29:14 | "In the RNN analysis we were passing the input the sequence so that's why we were able to preserve the sequence… but here we are going to pass the data in parallel — that's why we are not able to preserve the sequence of the data and we'll have to put this positional encoding over there." | Correct |
| 07:36:40–37:30 | "So this particular block will be able to see the entire sentence… it should not be like this… the training of the step-by-step model should learn the context of the sentence… this is going to be a cheating for the model. So what we do? So we hide the future word… we are giving one word at a time." | Correct — complete and accurate causal-masking explanation |
| 07:41:53–43:12 | "This query and key we are going to multiply, we are going to generate one score and we are going to normalize it using the divide of this under root k, means this is a dimension, and then we are going to perform the softmax, then this softmax we are going to multiply this value vector." | Correct — including `√d_k` |
| 07:42:44–42:49 | "This is the random weights… it's a trainable parameter." | Correct (W_Q, W_K, W_V are learned) |
| 07:45:25–47:28 | Four reasons RNN/LSTM was not right for fine-tuning: no parallelism/slow training; poor long-term dependency; no universal architecture ("someone was saying bidirectional, someone was saying ST, someone was saying encoder decoder with attention… no community standard"); no pretrained model. | **All four correct.** This is the module's core answer |
| 07:47:36–48:26 | "The rise of BERT and GPT destroyed the entire hype of RNN… the best example of it is Hugging Face. So many models, hundreds and thousands of models." | Correct (transcribed as "bird and GPT") |
| 06:11:44–11:58 | "We even we have a mamba also… it is a updated, upgraded version of the transformer… and more than transformer and maybe it could be the future." | **Wrong** — Mamba is an SSM, a different family. See §10.5 |

---

## Appendix B — Reference Links & Papers

| Paper | Year | Why it is in this module |
|---|---|---|
| Bengio, Simard & Frasconi — *Learning long-term dependencies with gradient descent is difficult* | 1994 | The original vanishing-gradient proof; the `λ^k` argument |
| Hochreiter & Schmidhuber — *Long Short-Term Memory* | 1997 | The LSTM; the constant error carousel |
| Sutskever, Vinyals & Le — *Sequence to Sequence Learning with Neural Networks* | 2014 | The encoder–decoder; the `(h, C)` bottleneck |
| Cho et al. — *Learning Phrase Representations using RNN Encoder–Decoder* | 2014 | The GRU; the second encoder–decoder paper of the same year |
| Bahdanau, Cho & Bengio — *Neural Machine Translation by Jointly Learning to Align and Translate* | 2014/2015 | Additive attention; the fix for the bottleneck; the ancestor of the transformer |
| Vaswani et al. — *Attention Is All You Need* | 2017 | The transformer; the ablations in §16.4 |
| Howard & Ruder — *Universal Language Model Fine-tuning for Text Classification* (ULMFiT) | 2018 | The proof that pretrain→fine-tune works on an LSTM; STLR, discriminative LRs, gradual unfreezing |
| Devlin et al. — *BERT* | 2018 | The first architecture that made this module's problems go away |
| Radford et al. — *GPT / GPT-2* | 2018/2019 | Pre-LN; the decoder-only recipe every 2026 LLM inherits |
| Khandelwal et al. — *Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context* | 2018 | The measured 50/200-token effective context — the real answer to §10.1 |
| Merity et al. — *Regularizing and Optimizing LSTM Language Models* (AWD-LSTM) | 2017 | The ULMFiT backbone; 57.3 perplexity on PTB |
| Su et al. — *RoFormer: Enhanced Transformer with Rotary Position Embedding* | 2021 | RoPE |
| Press, Smith & Lewis — *Train Short, Test Long: Attention with Linear Biases* (ALiBi) | 2021 | ALiBi |
| Xiong et al. — *On Layer Normalization in the Transformer Architecture* | 2020 | Why Pre-LN removes the warmup requirement |
| Zhang & Sennrich — *Root Mean Square Layer Normalization* | 2019 | RMSNorm |
| Shazeer — *GLU Variants Improve Transformer* | 2020 | SwiGLU; the `(8/3)·d_model` FFN sizing in §4.6.2 |
| Ainslie et al. — *GQA: Training Generalized Multi-Query Transformer Models* | 2023 | The 4–8× KV-cache reduction in §4.6.3 |
| Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* | 2022 | `O(n)` memory, 2–4× wall clock; §16.2 |
| Dao — *FlashAttention-2* | 2023 | ~2× FA1, 50–73 % of A100 peak |
| Shah et al. — *FlashAttention-3* | 2024 | Hopper warp-specialisation, FP8, ~75 % of peak |
| Gu, Goel & Ré — *Efficiently Modeling Long Sequences with Structured State Spaces* (S4) | 2021 | The modern SSM; HiPPO init |
| Gu & Dao — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* | 2023 | Selective SSM (S6); the parallel scan; 5× generation throughput |
| Dao & Gu — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2) | 2024 | The SSM↔attention bridge; semiseparable masks |
| Jelassi et al. — *Repeat After Me: Transformers are Better than State Space Models at Copying* | 2024 | The formal separation that limits pure SSMs |
| Lieber et al. — *Jamba: A Hybrid Transformer-Mamba Language Model* | 2024 | The 1:7 hybrid; 256 K context on one 80 GB GPU |
| Voita et al. — *Analyzing Multi-Head Self-Attention* | 2019 | Head specialization and pruning |
| Olsson et al. — *In-context Learning and Induction Heads* | 2022 | Induction heads; the phase change |
| Jain & Wallace — *Attention is not Explanation* | 2019 | Why attention weights are not evidence |

---

*End of CS-05. Pairs with `CH-05-RNN-LSTM-Transformers.md` (equations reference) and `IQ-05-RNN-LSTM-Transformers.md` (interview bank).*
