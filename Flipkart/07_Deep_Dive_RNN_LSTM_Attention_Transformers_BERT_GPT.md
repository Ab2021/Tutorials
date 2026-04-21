# 🧠 Deep Dive: RNN → LSTM → Attention → Transformers → BERT → GPT → Inference
### Progressive Technical Mastery for Flipkart Senior Data Scientist Interview

> **How to read this:** Each section builds on the previous.
> Start at RNNs even if you "know" them — the framing of WHY each innovation came is critical.
> Alex (PhD, CNN/DL background) can go very deep. Know the math, the intuition, AND the limitation at each step.

---

## THE JOURNEY AT A GLANCE

```
PROBLEM: "Process sequences of variable length, capturing long-range dependencies"

1950s  →  Turing / Automata (theoretical sequence processing)
1980s  →  RNN (recurrent neural network): process sequences step-by-step
1997   →  LSTM: solve vanishing gradient in RNNs
2014   →  Seq2Seq: encoder-decoder for translation
2015   →  Attention: "look back at all inputs while decoding"
2017   →  Transformer (Attention is All You Need): ditch the RNN entirely
2018   →  BERT: bidirectional transformer encoder, pre-trained at scale
2018   →  GPT: autoregressive transformer decoder, pre-trained at scale
2022+  →  GPT-3/4, instruction tuning, RLHF, efficient inference
```

---

# CHAPTER 1: THE PROBLEM — WHY DO WE NEED SEQUENCE MODELS?

### 1.1 What is a Sequence?

A **sequence** is data where order matters and elements are interdependent.

```
Examples:
├── Text: "The cat sat on the mat" — "mat" meaning depends on prior words
├── Time series: Stock price at t depends on t-1, t-2, ...
├── Audio: Speech waveform — each sample depends on prior samples
├── Claims text: "Patient was NOT in pain" — "not" changes everything backward
└── User behavior: click → add-to-cart → purchase — order defines intent
```

### 1.2 Why Standard Feed-Forward Networks Fail on Sequences

A standard **MLP (Multi-Layer Perceptron)** has a fundamental problem with sequences:

```
MLP architecture:
Input x (fixed size) → Hidden layers → Output y (fixed size)

Problems:
1. FIXED INPUT SIZE: "The cat sat" vs "The enormous fluffy cat sat" → different lengths
2. NO MEMORY: Each input processed independently — no notion of "previous words"
3. NO PARAMETER SHARING: Learning "cat" as subject at position 1 doesn't help at position 5

A model that processes "cat" at position 1 and position 5 should use
the same knowledge about what "cat" means — but MLP has separate weights per position.
```

**The requirement:** A model that:
- Handles variable-length inputs ✓
- Maintains a "memory" of what came before ✓
- Shares parameters across positions (same weights process any position) ✓

---

# CHAPTER 2: RNN — RECURRENT NEURAL NETWORK

### 2.1 Core Idea: Hidden State as Memory

An RNN processes one element at a time, maintaining a hidden state vector `h` that
acts as a "memory" of everything seen so far.

```
At each time step t:

Input:  x_t (current word/token embedding)
State:  h_{t-1} (memory from previous step)
Output: h_t (new memory) + y_t (prediction if needed)

FORMULA:
h_t = tanh(W_hh · h_{t-1}  +  W_xh · x_t  +  b_h)
       ↑                         ↑
   recurrent weight            input weight
   (same for all t)            (same for all t)

y_t = W_hy · h_t + b_y   (if predicting at each step)
```

**Visual unrolled RNN:**
```
x_1 ──→ [RNN] ──h_1──→ [RNN] ──h_2──→ [RNN] ──h_3──→ [RNN] ──h_4──→ y
         ↑               ↑               ↑               ↑
         W (shared)      W (shared)      W (shared)      W (shared)
```

**Key insight:** W_hh and W_xh are **shared across all time steps** — this is the parameter sharing we needed.

### 2.2 Training RNN: Backpropagation Through Time (BPTT)

To train an RNN, we unroll it through time and apply backpropagation:

```
FORWARD PASS: left to right, h_0 → h_1 → h_2 → ... → h_T
LOSS: L = Σ_t L(y_t, ŷ_t)
BACKWARD PASS: right to left (through time), compute gradients

Gradient of loss w.r.t. W_hh (accumulated across all steps):
∂L/∂W_hh = Σ_t ∂L_t/∂W_hh

∂L_t/∂h_k = (∂L_t/∂h_t) · Π_{j=k+1}^{t} (∂h_j/∂h_{j-1})
              ↑                              ↑
           loss gradient              product of Jacobians
           at time t                  from t back to k
```

### 2.3 The Vanishing Gradient Problem (WHY RNNs FAIL)

The key term is the **product of Jacobians**:
```
∂h_j/∂h_{j-1} = diag(tanh'(·)) · W_hh
```

If we chain this over T steps:
```
Π_{j=k+1}^{t} W_hh^T ≈ (λ_max)^(t-k)

If λ_max < 1  → gradients VANISH (exponentially small after ~10 steps)
If λ_max > 1  → gradients EXPLODE
```

**Practical consequence:**

```
Sentence: "The cat, who was old and grey and had lived for many years, ____ "
          ↑                                     long distance                 ↑
          subject                                                          verb must agree with subject
          (singular)                                                      (singular: "was")

RNN signal from "cat" at step 1 must survive 15 steps → VANISHES
RNN essentially "forgets" the subject by the time it needs it
```

**Solutions tried (and their limits):**
- Gradient clipping (for explosion only; doesn't fix vanishing)
- Careful weight initialization (helps slightly; not sufficient)
- Shorter sequences (not practical)
- → Need a fundamentally new architecture: **LSTM**

---

# CHAPTER 3: LSTM — LONG SHORT-TERM MEMORY

> Hochreiter & Schmidhuber, 1997 — one of the most cited papers in DL history

### 3.1 Core Idea: Explicit Memory Cell with Gates

LSTMs introduce a **cell state** `c_t` — a "conveyor belt" that information can flow along
with minimal modification. Information is added or removed via **gates** (learned sigmoid functions).

```
LSTM has TWO hidden states:
├── h_t: hidden state (short-term, passed to output and next cell)
└── c_t: cell state (long-term memory, the "conveyor belt")

THREE GATES (each is a sigmoid → output between 0 and 1):
├── Forget Gate f_t: How much of past cell state to KEEP
├── Input Gate i_t: How much of new information to ADD
└── Output Gate o_t: How much of cell state to OUTPUT
```

### 3.2 LSTM Equations (The Full Math)

At each time step t, given input x_t and previous hidden state h_{t-1}:

```
STEP 1: FORGET GATE — what to erase from cell state
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
      ↑
   sigmoid → values 0 to 1 per cell dimension
   0 = "forget everything", 1 = "remember everything"

STEP 2: INPUT GATE — what new info to write
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     ← how much to write
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   ← what to write (candidate values)

STEP 3: UPDATE CELL STATE — the core long-term memory update
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
       ↑                    ↑
   keep old info        add new info
   (element-wise)       (element-wise)

STEP 4: OUTPUT GATE — what to expose from cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
      ↑
   apply tanh to squash cell state, then gate the output
```

### 3.3 Why LSTM Solves Vanishing Gradients

The key is in the cell state update:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

The gradient flowing back through c_t to c_{t-1} is:
```
∂c_t/∂c_{t-1} = f_t   (element-wise — no matrix multiplication!)
```

Compare:
- **RNN:** gradient flows through W_hh^T at EVERY step → eigenvalue explosion/vanishing
- **LSTM:** gradient flows through f_t (forget gate values between 0 and 1) → additive, not multiplicative!

```
GRADIENT HIGHWAY: c_T → c_{T-1} → ... → c_1
Each step multiplies by f_t (a number between 0-1)
But the PATH EXISTS — the gradient can flow hundreds of steps without vanishing

The forget gate f_t ≈ 1 → gradient flows freely (model learns to "keep" important info)
The forget gate f_t ≈ 0 → gradient is blocked (model learned to "forget" — intentional)
```

### 3.4 GRU: Gated Recurrent Unit (Simplified LSTM)

GRU (Cho et al., 2014) simplifies LSTM by merging c_t and h_t into one state, using 2 gates instead of 3:

```
RESET GATE r_t = σ(W_r · [h_{t-1}, x_t])  ← how much past to use in candidate
UPDATE GATE z_t = σ(W_z · [h_{t-1}, x_t]) ← how much to update hidden state

h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])     ← candidate hidden state
h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  ← interpolate old and new

Advantages: Fewer parameters (3 weight matrices vs 4), faster training
Disadvantages: Slightly less expressive; performance usually similar to LSTM
```

### 3.5 Remaining Limitations of RNNs/LSTMs

Even with LSTM, fundamental problems remain:

```
PROBLEM 1: SEQUENTIAL COMPUTATION — cannot parallelize
LSTMs compute h_1 → h_2 → h_3 → ... sequentially
Cannot compute h_3 until h_2 is done → training is SLOW
For a 512-word document: 512 sequential steps before processing is done

PROBLEM 2: INFORMATION BOTTLENECK
For machine translation: encode full French sentence → single h_T vector → decode English
All information must pass through one vector → the bottleneck compresses everything
Long sentences: h_T can't remember the beginning

PROBLEM 3: FIXED CONTEXT WINDOW
When generating token t, the LSTM's "memory" of token k (far back) is still degraded
Even with LSTM, very long-range dependencies are hard

→ Solution: Add ATTENTION to allow decoder to look back at ALL encoder states
```

---

# CHAPTER 4: SEQ2SEQ + ATTENTION MECHANISM

### 4.1 The Seq2Seq Problem

Machine translation: "I love dogs" (English) → "J'aime les chiens" (French)

```
ENCODER: Reads input sequence, produces h_1, h_2, ..., h_T (one per input token)
DECODER: Generates output sequence token by token

WITHOUT ATTENTION (original Seq2Seq, Sutskever 2014):
Encoder: x_1 → x_2 → x_3 → h_3 (context vector)
Decoder: h_3 → y_1 → y_2 → y_3 (decoded output)

PROBLEM: All input information compressed into ONE vector h_3
         For long sentences, this context vector is an information bottleneck
```

### 4.2 Bahdanau Attention (2014) — The Breakthrough

> Bahdanau, Cho, Bengio — "Neural Machine Translation by Jointly Learning to Align and Translate"

**Core idea:** When generating each output word, let the decoder **attend to all encoder hidden states**,
weighted by how relevant each input position is to the current output.

```
ATTENTION MECHANISM:

At decoder step t, we have:
├── Decoder hidden state: s_t (what we're trying to decode)
└── Encoder hidden states: h_1, h_2, ..., h_T (all input positions)

STEP 1: Compute alignment scores (how relevant is input j to output t?)
e_{t,j} = score(s_t, h_j)
          ↑
      alignment model (learned neural network)
      Bahdanau: e_{t,j} = v^T · tanh(W_s · s_t + W_h · h_j)

STEP 2: Normalize scores to weights (sum to 1)
α_{t,j} = exp(e_{t,j}) / Σ_k exp(e_{t,k})
           ↑
       softmax → attention weights (how much to attend to position j)

STEP 3: Compute context vector (weighted sum of encoder states)
c_t = Σ_j α_{t,j} · h_j
      ↑
  weighted combination of ALL encoder states
  positions most relevant to current output get highest weight

STEP 4: Use context in decoder
s_t = f(s_{t-1}, y_{t-1}, c_t)   ← decoder LSTM cell uses context
y_t = g(s_t, c_t)                ← output prediction

A_{t,j} heatmap rows = output positions, columns = input positions
Bright cell = decoder attends strongly to that input when generating this output
```

**Why this is powerful:**
```
Generating "chiens" (dogs) → model attends strongly to "dogs" in input
Generating "J'" (I)        → model attends strongly to "I" in input
Even if input was 100 words long — no bottleneck!
```

### 4.3 Luong Attention (2015) — Simplification

Luong et al. proposed simpler attention that became more widely used:

```
SCORE FUNCTIONS (choose one):
├── Dot product:    score(s, h) = s^T · h           (fastest, no params)
├── General:        score(s, h) = s^T W_a h          (with learnable W_a)
└── Concat:         score(s, h) = v^T tanh(W_a[s;h]) (Bahdanau style)

KEY DIFFERENCE FROM BAHDANAU:
Bahdanau: compute attention before decoder LSTM step (additive)
Luong:    compute attention after decoder LSTM step (multiplicative)
```

---

# CHAPTER 5: SELF-ATTENTION — THE CORE OF TRANSFORMERS

### 5.1 From Attention to Self-Attention

In Bahdanau, attention is between two different sequences (encoder states → decoder query).

**Self-attention:** Apply attention within a SINGLE sequence.
Each position attends to all other positions in the SAME sequence.

```
Input: "The animal didn't cross the street because it was too tired"

Question: What does "it" refer to?
Self-attention allows position of "it" to attend to ALL other positions
→ High attention weight between "it" and "animal" → model knows "it" refers to animal

This is CONTEXT UNDERSTANDING — the same mechanism transformers use for everything
```

### 5.2 Scaled Dot-Product Attention — The Core Formula

```
INPUT:
├── Q (Query):  "What am I looking for?"        shape: [seq_len, d_k]
├── K (Key):    "What does each position offer?" shape: [seq_len, d_k]
└── V (Value):  "What does each position contain?" shape: [seq_len, d_v]

FORMULA:
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
                      ↑           ↑                 ↑
                  attention    scale factor     weighted values
                  weights      (prevents
                               vanishing
                               gradients in
                               large dims)

STEP BY STEP:
1. Q · K^T   → raw similarity scores  [seq_len × seq_len]
2. / √d_k    → scale (if d_k=64, divide by 8; prevents softmax saturation)
3. softmax() → attention weights       [seq_len × seq_len] (each row sums to 1)
4. · V       → weighted sum of values  [seq_len × d_v]
```

**Why √d_k scaling?**
```
Without scaling: for large d_k, dot products grow large in magnitude
→ softmax becomes extremely peaked (close to one-hot)
→ gradients become tiny (vanish)

With √d_k scaling: keeps dot product variance ≈ 1 regardless of d_k
→ softmax remains spread out → better gradient flow
```

### 5.3 Where Q, K, V Come From

In self-attention on the same input X:

```
X: input matrix [seq_len × d_model]
Q = X · W_Q   (W_Q: [d_model × d_k])
K = X · W_K   (W_K: [d_model × d_k])
V = X · W_V   (W_V: [d_model × d_v])

These are learned linear projections — same input X, projected into 3 different spaces
W_Q, W_K, W_V are learned parameters (trained via backprop)
```

**Intuition:**
```
W_Q: "What question should this token ask?"
W_K: "What key/tag should this token expose?"
W_V: "What content should this token share if attended to?"

"The" might expose:
  Q: "What do I modify?" → attends to nouns
  K: "I'm a definite article" → nouns attend to me
  V: "I am a function word, not much content"

"dog" might expose:
  Q: "What properties do I have? What verb applies to me?"
  K: "I'm a concrete noun" → pronouns, verbs attend to me
  V: "Rich semantic content about dogs"
```

---

# CHAPTER 6: MULTI-HEAD ATTENTION (MHA)

### 6.1 Why Multiple Heads?

Single attention computes ONE set of attention weights — one "relationship pattern" per input.

But language has MULTIPLE simultaneous relationships:
```
"The bank can guarantee deposits will eventually cover future tuition costs"

Head 1 (syntactic): "bank" → "can" (subject-verb relationship)
Head 2 (semantic):  "bank" → "deposits" (financial entity relationship)
Head 3 (coreference): No pronoun here, but detects that "costs" is the object of "cover"
Head 4 (distance):  Short-range syntactic agreement patterns

One head cannot capture all of these simultaneously
→ Multi-head: run H parallel attention heads, each can specialize
```

### 6.2 Multi-Head Attention — Full Math

```
For head i (i = 1, 2, ..., H):
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

Each head has separate projection matrices:
W_Q_i: [d_model × d_k],  d_k = d_model / H
W_K_i: [d_model × d_k]
W_V_i: [d_model × d_v],  d_v = d_model / H

CONCATENATE all heads:
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) · W_O

W_O: [H·d_v × d_model]  ← output projection back to d_model dimensions

PARAMETER COUNT:
Per head: 3 projection matrices W_Q, W_K, W_V + one output W_O
Total: 4 × d_model × d_model parameters (with H heads of size d_model/H each)
```

**Standard configuration (as in original Transformer paper):**
```
d_model = 512
H = 8 heads
d_k = d_v = 64 per head
```

### 6.3 What Each Head Learns (Empirically)

Research (Voita et al. 2019, Clark et al. 2019) shows heads specialize:

```
Positional heads:    Attend to immediately adjacent tokens (syntax)
Syntactic heads:     Attend to syntactic dependents (subject → verb)
Rare word heads:     Attend strongly to rare/unusual tokens
Co-reference heads:  Attend to pronouns' antecedents
Delimiter heads:     Attend to [SEP] / punctuation (sentence boundaries)
Semantic heads:      Attend to semantically related words regardless of distance
```

---

# CHAPTER 7: THE TRANSFORMER ARCHITECTURE

> Vaswani et al., 2017 — "Attention Is All You Need"
> The paper that changed everything: NO recurrence, NO convolutions — ONLY attention

### 7.1 Overall Architecture

```
TRANSFORMER (for Seq2Seq, e.g., translation):

ENCODER STACK (N=6 layers):
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│  Encoder Layer × 6:                 │
│  ┌────────────────────────────────┐ │
│  │  Multi-Head Self-Attention     │ │
│  │  Add & Layer Norm              │ │
│  │  Feed-Forward Network          │ │
│  │  Add & Layer Norm              │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓ (encoder output: K, V for cross-attention)

DECODER STACK (N=6 layers):
Output Tokens (shifted right)
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│  Decoder Layer × 6:                 │
│  ┌────────────────────────────────┐ │
│  │  MASKED Self-Attention         │ │  ← causal masking
│  │  Add & Layer Norm              │ │
│  │  Cross-Attention               │ │  ← Q from decoder, K,V from encoder
│  │  Add & Layer Norm              │ │
│  │  Feed-Forward Network          │ │
│  │  Add & Layer Norm              │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
Linear + Softmax → output token probabilities
```

### 7.2 Key Components in Detail

#### Positional Encoding — Giving Position Information

Self-attention is PERMUTATION INVARIANT by default:
```
"dog bit man" and "man bit dog" → same attention weights if not for position info
We need to inject position information

ORIGINAL TRANSFORMER: Sinusoidal Positional Encoding (fixed, not learned)
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

WHY SINUSOIDAL?
- Each position gets a unique encoding
- The model can learn to attend to RELATIVE positions:
  PE(pos+k) can be expressed as linear function of PE(pos)
  → relative position patterns are learnable

MODERN PRACTICE: Learnable positional embeddings (like BERT uses)
  or RoPE (Rotary Position Embedding) for better length generalization (LLaMA, GPT-NeoX)
```

#### Add & Layer Norm — Residual Connections

```
RESIDUAL CONNECTION (He et al. 2015 from ResNet):
Output = LayerNorm(x + Sublayer(x))
          ↑              ↑
    normalize      residual: add input x
                   to sublayer output

WHY RESIDUALS?
- Gradient flows directly back through the skip connection
- Allows training very deep networks (original transformer has 6+6=12 layers)
- Initializing as identity: network learns to add, not transform from scratch

WHY LAYER NORM (not Batch Norm)?
- Batch Norm normalizes across the batch dimension → fails for sequences (variable length)
- Layer Norm normalizes across the feature dimension within each example → works for any length
- LN(x) = (x - μ) / σ · γ + β  (per-example, per-layer normalization)
```

#### Feed-Forward Network (FFN)

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
          ↑
       ReLU activation (or GELU in modern variants)

Dimensions: d_model → d_ff → d_model  (d_ff = 4 × d_model typically)
d_model=512: 512 → 2048 → 512

WHY FFN AFTER ATTENTION?
Attention mixes information between positions but applies same transformation to all
FFN applies position-wise transformation — increases expressivity per token
"Attention = routing/communication; FFN = computation/processing"
```

#### Causal (Masked) Self-Attention in Decoder

```
PROBLEM: During training, we have the full target sequence.
But the model must NOT look at future tokens (it must predict them).

SOLUTION: Masking — set future attention scores to -∞ before softmax

Attention mask (for seq_len=4):
         pos_0  pos_1  pos_2  pos_3
pos_0  [  0     -∞     -∞     -∞  ]
pos_1  [  0      0     -∞     -∞  ]
pos_2  [  0      0      0     -∞  ]
pos_3  [  0      0      0      0  ]

Add this mask to Q·K^T before softmax:
-∞ → softmax(−∞) = 0 → zero attention weight → future tokens are invisible

This is CAUSAL MASKING — position t can only attend to positions ≤ t
```

### 7.3 Complexity Analysis — Why Transformers Scale Better

```
                    RNN         Transformer
Sequential ops:     O(n)        O(1)          ← parallelizable!
Max path length:    O(n)        O(1)          ← any token can attend to any token
Computation:        O(n · d²)   O(n² · d)     ← quadratic in sequence length
Memory:             O(n · d)    O(n² + n · d)

Trade-off: Transformers are quadratic in sequence length (attention matrix n×n)
           but parallelizable → much faster on GPU hardware
```

---

# CHAPTER 8: BERT — BIDIRECTIONAL ENCODER REPRESENTATIONS FROM TRANSFORMERS

> Devlin et al., 2018 (Google) — revolutionized NLP with pre-training

### 8.1 The Pre-Training Paradigm

Before BERT, NLP models were trained from scratch on small labeled datasets.
BERT established: **pre-train on massive unlabeled text → fine-tune on specific tasks**

```
PRE-TRAINING: 
├── Data: Wikipedia (2.5B words) + BookCorpus (800M words)
├── Process: Unsupervised → model learns language representations
└── Result: General-purpose language understanding model

FINE-TUNING:
├── Take pre-trained BERT weights
├── Add task-specific head (classification layer, span predictor, etc.)
├── Train on small labeled dataset (few thousand examples)
└── Result: State-of-the-art on specific task with minimal task-specific data
```

### 8.2 BERT's Architecture — Encoder Only

BERT uses ONLY the Transformer ENCODER stack (no decoder):

```
BERT-Base:  12 encoder layers, d_model=768, 12 attention heads = 110M parameters
BERT-Large: 24 encoder layers, d_model=1024, 16 attention heads = 340M parameters

KEY DIFFERENCE FROM DECODER:
BERT encoder uses FULL self-attention (no causal masking)
→ Each position attends to ALL other positions (both left and right)
→ Bidirectional context (can see "it" AND what comes after to resolve coreference)
→ NOT suitable for generation (sees the answer while predicting it)
→ IDEAL for understanding tasks: classification, NER, QA
```

### 8.3 BERT's Special Tokens

```
[CLS] token: Classification token, always at position 0
             Final hidden state of [CLS] used as sentence representation for classification

[SEP] token: Separator between sentences or segments
             Used for sentence pair tasks (NLI, QA)

[MASK] token: Placeholder for masked words during pre-training

Example input:
[CLS] The patient was admitted [MASK] the hospital . [SEP] The diagnosis was unclear . [SEP]
  ↑                               ↑                    ↑
CLS vector                    masked word           segment separator
(for classification)          for MLM               (sentence B starts)
```

### 8.4 BERT's Two Pre-Training Objectives

#### Objective 1: Masked Language Modeling (MLM)

```
TASK: Randomly mask 15% of tokens → predict the masked tokens

For each selected token (15% of all tokens):
├── 80% of the time: replace with [MASK]
├── 10% of the time: replace with a random word
└── 10% of the time: keep the original word

WHY NOT JUST 100% [MASK]?
- Fine-tuning has NO [MASK] tokens → train/test mismatch
- 10% random: forces model not to blindly trust context (can't always trust surrounding words)
- 10% original: forces model to have representation for all tokens, not just masked ones

LOSS: Cross-entropy on the masked tokens only:
L_MLM = -Σ_{masked t} log P(x_t | x_{masked context})

KEY PROPERTY: Bidirectional — model can use LEFT and RIGHT context to predict mask
"The [MASK] sat on the mat" → uses "The", "sat", "on", "the", "mat" → predicts "cat"
```

#### Objective 2: Next Sentence Prediction (NSP)

```
TASK: Given two sentences A and B, predict whether B follows A in the original text

50%: B is the actual next sentence → label: IsNext
50%: B is a random sentence → label: NotNext

Input: [CLS] sentence_A [SEP] sentence_B [SEP]
Output: Binary classification using [CLS] vector

WHY? Tasks like QA and NLI require understanding sentence-level relationships

NOTE: Later research (RoBERTa, 2019) showed NSP actually HURTS performance slightly
→ RoBERTa drops NSP, uses longer training → outperforms BERT
```

### 8.5 BERT's Embeddings

```
BERT input = Token Embedding + Segment Embedding + Position Embedding

Token Embedding: 30,000 vocabulary (WordPiece tokenization)
                "playing" → ["play", "##ing"] → two tokens
Segment Embedding: 0 for sentence A, 1 for sentence B
Position Embedding: LEARNED (not sinusoidal like original transformer), max 512

All three embeddings are ADDED element-wise → input to first encoder layer
```

### 8.6 Fine-Tuning BERT for Downstream Tasks

```
TASK 1: Text Classification (sentiment, topic):
[CLS] → [768-dim vector] → Linear(768, num_classes) → softmax
Train: BERT weights + classification head jointly

TASK 2: Named Entity Recognition (token classification):
Each token embedding → Linear(768, num_labels) → softmax per token
Output: B-PER, I-PER, O, B-ORG, ... for each token

TASK 3: Question Answering (SQuAD):
Input: [CLS] Question [SEP] Context [SEP]
Output: Two vectors (Start, End) over context positions → span extraction
Linear(768, 1) per token → argmax = answer start/end position

TASK 4: Sentence Similarity (GLUE):
[CLS] sentence_A [SEP] sentence_B [SEP] → [CLS] vector → regression/classification
```

---

# CHAPTER 9: GPT — GENERATIVE PRE-TRAINED TRANSFORMER

> Radford et al., 2018 (OpenAI) — the other revolutionary pre-training approach

### 9.1 GPT vs BERT: The Fundamental Philosophy Difference

```
                BERT                        GPT
Architecture:   Encoder only               Decoder only
Attention:      Bidirectional              Causal (left-to-right only)
Pre-training:   Masked LM + NSP            Causal Language Modeling
Goal:           UNDERSTAND text            GENERATE text
Strength:       Classification, NER, QA    Text generation, few-shot learning
Weakness:       Can't generate naturally   Lower performance on understanding tasks
```

### 9.2 GPT's Pre-Training Objective: Causal Language Modeling

```
TASK: At each position t, predict the next token given all previous tokens
P(x_t | x_1, x_2, ..., x_{t-1}) for all t

LOSS: Standard cross-entropy (negative log-likelihood):
L = -Σ_t log P(x_t | x_{<t})

This is EXACTLY the language modeling objective — the model learns to continue text

WHY CAUSAL?
Generation is inherently left-to-right → causal masking enforces this
BERT's bidirectional context is useless for generation (cheating to see future tokens)
```

### 9.3 GPT Architecture: Decoder Stack Only

```
GPT-1:  12 decoder layers, d_model=768, 12 heads, 117M parameters
GPT-2:  48 decoder layers, d_model=1600, 25 heads, 1.5B parameters
GPT-3:  96 decoder layers, d_model=12288, 96 heads, 175B parameters
GPT-4:  ~1T+ parameters, multi-modal (estimated, not confirmed)

Each Decoder Layer:
├── Causal (Masked) Multi-Head Self-Attention
├── Add & Layer Norm
├── Feed-Forward Network
└── Add & Layer Norm

MODIFICATIONS from original Transformer decoder:
- No cross-attention (no encoder to cross-attend to)
- Layer norm BEFORE attention (Pre-LN, not Post-LN) in GPT-2+ → more stable training
```

### 9.4 GPT-3's Key Innovation: In-Context Learning (Few-Shot)

```
GPT-3 demonstrated that LARGER MODELS can learn from examples in the prompt
WITHOUT any gradient updates — just from examples in the context.

ZERO-SHOT:
"Translate English to French: 'The cat sat on the mat' →"

ONE-SHOT:
"Translate English to French: 'Hello' → 'Bonjour'. 'The cat sat on the mat' →"

FEW-SHOT:
"English: 'Hello' → French: 'Bonjour'
 English: 'Goodbye' → French: 'Au revoir'
 English: 'The cat sat on the mat' → French:"

The model "learns" the task from examples in context — no weight updates!
This is ICL (In-Context Learning) — emergent capability at large scale
```

### 9.5 The Scaling Laws (Kaplan et al., 2020 — OpenAI)

Critical for understanding why GPT keeps getting bigger:

```
Model performance (cross-entropy loss) follows power laws with:
├── Model parameters N: L ∝ N^{-0.076}
├── Dataset tokens D: L ∝ D^{-0.095}
└── Compute FLOPs C: L ∝ C^{-0.050}

ALL THREE matter — scaling any one alone gives diminishing returns.

CHINCHILLA INSIGHT (Hoffmann et al., 2022 — DeepMind):
To optimally use compute budget C:
  N_opt = (C / 6)^{0.5}    (model parameters)
  D_opt = (C / 6)^{0.5}    (training tokens)
  → N and D should scale EQUALLY

Example: GPT-3 was UNDERTRAINED for its size (175B params, ~300B tokens)
         Chinchilla-optimal: 175B params → needs 3.5T tokens
```

---

# CHAPTER 10: VARIANTS, IMPROVEMENTS & MODERN MODELS

### 10.1 RoBERTa (Liu et al., 2019 — Facebook)

```
"A Robustly Optimized BERT Pretraining Approach" — same architecture, better training:

1. REMOVE NSP: Drop Next Sentence Prediction — hurts, not helps
2. LONGER TRAINING: 10x more compute than BERT
3. BIGGER BATCHES: 8K sequences vs. 256 (better gradient estimates)
4. MORE DATA: 160GB of text vs. BERT's 16GB
5. DYNAMIC MASKING: Different mask each epoch vs. static mask
6. NO [SEP] between sentences: Simpler, longer sequences

Result: RoBERTa-Large beats BERT-Large on all GLUE benchmarks
Key lesson: Pre-training matters more than architecture details
```

### 10.2 T5 (Raffel et al., 2020 — Google)

```
"Text-to-Text Transfer Transformer"

PRINCIPLE: Frame EVERY NLP task as text-to-text:
  Classification: "Classify: this movie was great." → "positive"
  Translation:    "Translate to French: The cat sat." → "Le chat était assis."
  Summarization:  "Summarize: [long article]" → "short summary"
  QA:             "Question: What is X? Context: [passage]" → "the answer"

Architecture: Full encoder-decoder transformer
Training: Mixture of tasks + span masking (mask spans, not individual tokens)
Size: T5-11B = 11 billion parameters (2020 state-of-the-art)

KEY INSIGHT: Unified format → one model for all tasks → simpler deployment
```

### 10.3 LLAMA / LLAMA-2 / LLAMA-3 (Meta)

```
KEY INNOVATIONS over GPT-3:
1. RoPE (Rotary Position Embedding) instead of learned absolute positions
   → Better length generalization (trained at 4K, works at 8K+)
2. SwiGLU activation: swish(x) · gated_linear → better than ReLU/GELU
3. RMSNorm instead of LayerNorm: faster, similar performance
4. Grouped Query Attention (GQA) in LLaMA-2/3:
   Multiple query heads share K,V → reduces memory during inference

LLAMA-2 (70B) nearly matches GPT-3.5 with OPEN WEIGHTS
→ Enabled massive open-source LLM ecosystem (Mistral, Falcon, Vicuna, etc.)
```

### 10.4 Instruction Tuning & RLHF

```
RAW GPT: Good at continuing text, but doesn't follow instructions well
"Explain quantum physics" → might continue with "is a great topic for ..."
                          rather than actually explaining

INSTRUCTION TUNING (FLAN, InstructGPT):
Fine-tune on dataset of (instruction, response) pairs
→ Model learns to DO what the instruction says, not just continue text

RLHF (Reinforcement Learning from Human Feedback):
Step 1: Supervised Fine-Tuning (SFT) on demonstration data
Step 2: Train Reward Model (RM) on human preference data
         Given (prompt, response_A, response_B) + human preference → train RM
Step 3: PPO (Proximal Policy Optimization) to optimize LLM using RM reward

Result: ChatGPT-style helpful, harmless, honest behavior

DPO (Direct Preference Optimization, Rafailov 2023):
Skips Step 2 and 3 — directly optimizes on preference pairs
L_DPO = -log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
Simpler, more stable than PPO; often used in practice
```

---

# CHAPTER 11: INFERENCE METHODS — HOW TO GENERATE TEXT

### 11.1 The Decoding Problem

During inference, the model outputs a probability distribution over vocabulary at each step.
**Decoding strategy** determines HOW we convert these probabilities into actual tokens.

```
At step t, model outputs: P(token | prefix) for all ~50,000 vocabulary tokens

We must CHOOSE one token → this choice affects all future tokens (autoregressive)

Trade-off: Quality vs. Diversity
```

### 11.2 Greedy Decoding

```
STRATEGY: At each step, pick the HIGHEST probability token

x_t = argmax P(x_t | x_{<t})

EXAMPLE:
Step 1: "The" [0.3] "A" [0.25] "This" [0.2] ... → pick "The"
Step 2: "cat" [0.4] "dog" [0.2] ...             → pick "cat"
Step 3: "sat" [0.5] "ran" [0.3] ...             → pick "sat"

ADVANTAGE: Fast, deterministic, simple
DISADVANTAGE: Locally optimal ≠ globally optimal

EXAMPLE OF FAILURE:
                    "The" [0.3]  →  "cat" [0.5]   →  total: 0.15  ✓ picked
                    "A"   [0.25] →  "large" [0.8] →  total: 0.20  ← better!

Greedy picked "The cat" (0.15) but "A large ..." (0.20) is higher probability
→ Greedy decoding can get stuck in suboptimal local choices
```

### 11.3 Beam Search

```
STRATEGY: Keep K (beam width) candidate sequences at each step

EXAMPLE with beam_width = 2:

Step 1: Top 2 tokens:
  Beam 1: "The" [prob: 0.30]
  Beam 2: "A"   [prob: 0.25]

Step 2: For EACH beam, expand top 2 tokens:
  From "The": "cat" [0.30 × 0.50 = 0.15], "dog" [0.30 × 0.20 = 0.06]
  From "A":   "large" [0.25 × 0.80 = 0.20], "small" [0.25 × 0.30 = 0.075]
  Keep top 2: "A large" [0.20], "The cat" [0.15]

Step 3: Continue expanding from top 2 beams...

FINAL: Return the beam with highest cumulative log-probability

ADVANTAGE: Better than greedy (doesn't get stuck in local optima)
DISADVANTAGE: 
- Still deterministic
- Can produce repetitive, generic text
- Favors shorter sequences without length normalization
- Doesn't capture diversity

LENGTH NORMALIZATION:
score(seq) = log P(seq) / |seq|^α   (α typically 0.6-0.8)
Prevents model from preferring short sequences (which have fewer probability terms to multiply)
```

### 11.4 Sampling Methods

```
STRATEGY: Sample from the probability distribution instead of taking argmax
→ Non-deterministic → different outputs each call

RAW SAMPLING:
x_t ~ P(x_t | x_{<t})  (sample directly from full distribution)

PROBLEM: Includes very low probability tokens → incoherent outputs
"The cat sat on the" → might sample "refrigerator" [prob: 0.00001] → nonsense
```

#### Temperature Sampling

```
Modify the distribution by a temperature parameter T:

P_T(x_t) = softmax(logits / T)

T < 1 (e.g., 0.7):  Sharpen distribution → more deterministic, safer choices
T = 1:              Original distribution
T > 1 (e.g., 1.5):  Flatten distribution → more uniform → more random/creative

EFFECT ON OUTPUT:
T → 0: Approaches greedy (lowest entropy)
T → ∞: Approaches uniform random sampling (maximum entropy)

TYPICAL USE:
Creative writing: T = 0.8-1.0
Factual tasks:    T = 0.2-0.5
Code generation:  T = 0.0-0.2 (deterministic preferred)
```

#### Top-K Sampling

```
STRATEGY: At each step:
1. Take top K tokens by probability
2. Re-normalize to sum to 1
3. Sample from just these K tokens

EXAMPLE with K=5:
Full vocab: "cat" [0.4], "dog" [0.2], "bird" [0.1], ..., "refrigerator" [0.00001]
Top 5: "cat" [0.4], "dog" [0.2], "bird" [0.1], "fish" [0.08], "mouse" [0.05]
Re-normalize: "cat" [0.49], "dog" [0.24], ...
Sample from this → never pick low-probability garbage

PROBLEM: K is fixed regardless of distribution shape
Distribution A: [0.99, 0.005, 0.005]  → K=5: wasteful (top 1 dominates)
Distribution B: [0.1, 0.1, 0.09, ...] → K=5: too restrictive (many equally valid)
```

#### Top-P (Nucleus) Sampling

```
STRATEGY: At each step:
1. Sort tokens by probability (descending)
2. Take the SMALLEST set of tokens whose cumulative probability ≥ p
3. Re-normalize and sample

EXAMPLE with p=0.9:
"cat" [0.4] → cumulative: 0.4
"dog" [0.2] → cumulative: 0.6
"bird" [0.1] → cumulative: 0.7
"fish" [0.08] → cumulative: 0.78
"mouse" [0.07] → cumulative: 0.85
"hamster" [0.05] → cumulative: 0.90 ← STOP here (reached p=0.9)
Sample from {cat, dog, bird, fish, mouse, hamster}

ADVANTAGE: Dynamically adapts to distribution shape
High-confidence step (peaked): few tokens in nucleus (focused sampling)
Uncertain step (flat): many tokens in nucleus (creative sampling)

TYPICAL SETTINGS: p=0.9, temperature=0.8 → good balance of quality and diversity
```

#### Typical/Local Typical Sampling (2022)

```
NEW IDEA: Sample tokens that are "typical" — not too surprising, not too predictable
Based on information-theoretic surprise: -log P(x_t | x_{<t})

Tokens with surprise close to entropy H[P] are "typical"
Over-confident (low surprise) → boring
High surprise → incoherent

Used in some modern generation APIs; more complex to implement
```

### 11.5 Repetition Penalty

```
PROBLEM: Greedy and beam search generate repetitive text:
"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."

SOLUTION: Penalize recently generated tokens

Modified score: logit(x_t) = logit(x_t) / penalty_factor
                if x_t was generated in recent window

penalty_factor > 1 → reduces probability of repeating tokens
typical value: 1.1 - 1.3

More sophisticated: presence penalty (any repeat) vs. frequency penalty (penalize proportional to frequency)
Used in OpenAI API parameters
```

### 11.6 Speculative Decoding (Efficiency Innovation)

```
PROBLEM: Large LLMs are slow at inference (each token requires full forward pass through 175B params)

SPECULATIVE DECODING (Chen et al. 2023):
1. Small "draft" model (fast) generates K tokens speculatively
2. Large "target" model validates all K tokens in ONE parallel forward pass
3. Accept tokens that match target model distribution; reject others
4. Generates 2-4× speedup with IDENTICAL output distribution

┌─────────────────────────────────────────────────────────┐
│  Draft model (7B)   → generates "The cat sat on the"    │
│  Target model (70B) → validates all 5 tokens in parallel│
│  "The" ✓ "cat" ✓ "sat" ✓ "on" ✗ → reject "on" onward  │
│  Target model generates "slept" instead                  │
│  Net: 3 tokens validated + 1 corrected = 3/4 = 75% hit  │
└─────────────────────────────────────────────────────────┘

WHY IT WORKS: Large model forward pass over K tokens ≈ same cost as 1 token inference
(compute is dominated by model params, not sequence length for short sequences)
```

### 11.7 KV Cache — The Critical Inference Optimization

```
PROBLEM: Autoregressive generation recomputes K,V for ALL previous tokens every step

At step t, attention requires:
K_{1..t} = [X_{1..t} · W_K]   ← recomputing old K for positions 1..t-1 every time!
V_{1..t} = [X_{1..t} · W_V]

SOLUTION: KV Cache — store K,V tensors from previous steps
Step 1: Compute K_1, V_1 → cache
Step 2: Only compute K_2, V_2 (new token) → append to cache → use all [K_1,K_2], [V_1,V_2]
Step t: Only compute K_t, V_t → append → use cached [K_1..K_t], [V_1..V_t]

MEMORY COST:
KV cache size = 2 × num_layers × num_heads × seq_len × d_head × batch_size × 2 bytes
For LLaMA-2-70B at seq_len=4096: ~70GB per request!
→ KV cache is the main bottleneck for long-context LLMs

SOLUTIONS TO KV CACHE MEMORY:
├── Grouped Query Attention (GQA): Multiple Q heads share one K,V head
│   Reduces KV cache by num_heads / num_groups
├── Multi-Query Attention (MQA): Extreme: ALL Q heads share single K,V
│   Reduces KV cache by num_heads (e.g., 32x)
├── PagedAttention (vLLM): Virtual memory paging for KV cache
│   Reduces waste from over-provisioning; enables 2-4x throughput increase
└── Sliding Window Attention (Mistral): Only attend to last W tokens
    KV cache stays constant at W regardless of sequence length
```

---

# CHAPTER 12: QUICK REFERENCE — COMPARISON TABLE

```
┌──────────────┬────────────┬────────────┬────────────────┬─────────────────────┐
│ Model        │ Architecture│ Attention  │ Pre-training   │ Best For            │
├──────────────┼────────────┼────────────┼────────────────┼─────────────────────┤
│ RNN          │ Recurrent  │ None       │ Supervised     │ Short sequences      │
│ LSTM         │ Recurrent+ │ None       │ Supervised     │ Medium sequences     │
│ Seq2Seq      │ E+D LSTM   │ Bahdanau   │ Supervised     │ Translation (small)  │
│ Transformer  │ E+D Attn   │ Full MHA   │ Supervised     │ Translation (large)  │
│ BERT         │ Enc only   │ Bidirect'l │ MLM + NSP      │ Classification, NER  │
│ GPT-1/2/3    │ Dec only   │ Causal     │ Causal LM      │ Text generation      │
│ T5           │ E+D Attn   │ Full MHA   │ Span masking   │ Any NLP (unified)    │
│ RoBERTa      │ Enc only   │ Bidirect'l │ MLM (better)   │ Classification       │
│ LLaMA-2/3    │ Dec only   │ GQA+Causal │ Causal LM      │ Gen + instruction    │
│ GPT-4        │ Dec only   │ MHA+Causal │ RLHF           │ All tasks            │
└──────────────┴────────────┴────────────┴────────────────┴─────────────────────┘
```

---

# CHAPTER 13: FOR THE INTERVIEW — EXPECT THESE EXACT QUESTIONS

### Q1: "Explain BERT's attention mechanism through the forward pass"

**Start to finish answer:**

> "BERT takes a tokenized input sequence — say 'The cat sat' with [CLS] prepended and [SEP]
> at the end. Each token is embedded via the sum of token, segment, and position embeddings,
> giving a matrix X of shape [seq_len × 768].
>
> In each of the 12 encoder layers, X is projected into Q, K, V via learned weight matrices
> W_Q, W_K, W_V ∈ R^{768×64} for each of 12 heads.
>
> Self-attention computes: Attention(Q,K,V) = softmax(QK^T/√64)V
>
> The √64 scaling prevents the dot products from becoming too large (which would saturate
> the softmax and kill gradients). Each of the 12 heads computes this independently, then
> the outputs are concatenated and projected back to 768 dimensions via W_O.
>
> Critically, BERT has NO causal mask — every token can attend bidirectionally to every
> other token. This is why it's powerful for understanding but can't generate."

---

### Q2: "How does BERT differ from GPT? When would you use each?"

| | BERT | GPT |
|---|---|---|
| **Architecture** | Encoder only | Decoder only |
| **Attention** | Full bidirectional | Causal (left-to-right) |
| **Pre-training** | Masked LM + NSP | Causal language modeling |
| **Strengths** | Understanding, classification | Generation, few-shot |
| **Inference** | Single pass (fast for classification) | Autoregressive (slow) |
| **Use in fraud** | Extract features from claims text | Generate investigation summaries |

---

### Q3: "What is the vanishing gradient problem and how does LSTM solve it?"

> "In vanilla RNNs, gradients flow back through the chain rule as a product of Jacobians.
> Each Jacobian contains the weight matrix W_hh transposed — if its eigenvalues are < 1,
> this product shrinks exponentially over time steps. With 100 steps and eigenvalue 0.9:
> 0.9^100 ≈ 0.0000265 — the gradient has essentially vanished.
>
> LSTM solves this via the cell state c_t and the forget gate f_t. The cell state update is:
> c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
> The gradient of c_t with respect to c_{t-1} is simply f_t — element-wise multiplication,
> no matrix multiply! When the forget gate is near 1 (the model decides to remember),
> gradients flow essentially unchanged. This creates a 'gradient highway' through time."

---

### Q4: "Why does multi-head attention use √d_k scaling?"

> "When we compute QK^T, each element is a dot product of two d_k-dimensional vectors.
> If the components are independent with mean 0 and variance 1, each element has
> expected value 0 and variance d_k (sum of d_k terms each with variance 1).
> So the std dev of dot products scales as √d_k.
>
> Without scaling, for large d_k (e.g., 64 in BERT), dot products have std dev 8.
> After softmax, the distribution becomes extremely peaked — nearly one-hot.
> This means almost all attention goes to one position, gradients become tiny.
>
> Dividing by √d_k normalizes variance back to 1, keeping the softmax well-behaved
> and allowing gradients to flow through all positions."

---

### Q5: "What are the training objectives for BERT vs GPT?"

> "BERT: Masked Language Modeling (randomly mask 15% of tokens → predict them using
> BOTH left and right context) + Next Sentence Prediction. The key is that the loss
> is ONLY computed on masked tokens, not all tokens.
>
> GPT: Causal Language Modeling — predict the next token given all previous tokens —
> essentially minimizing negative log-likelihood over all token positions:
> L = -Σ_t log P(x_t | x_{<t}).
>
> The philosophical difference: BERT is discriminative — learn to fill in blanks.
> GPT is generative — learn to continue text. These different objectives lead to
> their different strengths: BERT understands context better; GPT generates more naturally."

---

### Q6: Flipkart-specific — "Which model would you use for semantic search in Flipkart's catalog?"

> "For semantic product search, I'd use a bi-encoder architecture with BERT:
> - Encode product descriptions offline → 768-dim embeddings stored in FAISS/Annoy index
> - Encode user query online → 768-dim embedding
> - Approximate nearest neighbor search → top-k semantically similar products
>
> I'd fine-tune with domain-specific data: (query, relevant_product, irrelevant_product)
> triplets, using contrastive learning (InfoNCE/SupCon loss) to bring query closer to
> relevant products and away from irrelevant ones in embedding space.
>
> For re-ranking the top-k: use a cross-encoder (full BERT attention over [query, product])
> which is more accurate but slower — acceptable for ranking 50 candidates.
>
> Two-stage: bi-encoder (fast, approximate) → cross-encoder (accurate, expensive).
> This is exactly the same pattern as my RAG fraud system at Chubb."

---

*End of Deep Dive: RNN → LSTM → Attention → Transformers → BERT → GPT → Inference*
