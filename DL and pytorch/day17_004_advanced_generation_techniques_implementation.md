# Day 17.4: Advanced Generation Techniques - A Practical Guide

## Introduction: Beyond Greedy Search

We have seen that a generative language model like GPT works by predicting a probability distribution for the next token and then sampling from it. The simplest way to sample is **greedy search**: at each step, we just choose the single token with the highest probability. 

While simple, greedy search often leads to repetitive, dull, and unnatural-sounding text. To generate more creative, coherent, and human-like text, we need more advanced decoding and sampling strategies.

This guide provides a practical overview of the most common and effective techniques used for controlling the output of large language models during generation.

**Today's Learning Objectives:**

1.  **Understand the Limitations of Greedy Search:** See why always choosing the most likely token is not optimal.
2.  **Learn about Beam Search:** Understand this technique for keeping track of multiple high-probability sequences at each step.
3.  **Explore Advanced Sampling Methods:** Learn how to use **Temperature**, **Top-k Sampling**, and **Nucleus (Top-p) Sampling** to control the randomness and creativity of the generated text.
4.  **Prevent Repetition:** See how to use penalties like `no_repeat_ngram_size` to improve text quality.
5.  **Use the Hugging Face `generate` Method:** Learn how to easily apply all these techniques using the powerful and flexible `.generate()` method in the `transformers` library.

---

## Part 1: The Problem with Greedy Search

Greedy search makes the locally optimal choice at each step. However, a series of locally optimal choices does not always lead to a globally optimal sequence. A high-probability word early on might lead to a dead-end, resulting in a low-probability sequence overall.

**Example:**
*   Prompt: "The best way to learn is to"
*   Greedy choice at step 1: "practice" (high probability)
*   Sequence so far: "The best way to learn is to practice"
*   Greedy choice at step 2: "practice" (the model might get stuck in a loop)
*   Result: "...practice practice practice..."

We need methods that explore a wider range of possibilities.

---

## Part 2: Beam Search

**The Idea:** Instead of just keeping the single best prediction at each step, **beam search** keeps track of the `k` most probable partial sequences (or "beams").

**The Process:**
1.  At the first step, generate the probability distribution for the next token and select the top `k` most likely tokens. These are your `k` initial beams.
2.  For the second step, for *each* of the `k` beams, predict the next token. This gives you `k * vocab_size` possible new sequences.
3.  Calculate the cumulative probability of all these new sequences.
4.  Select the top `k` most probable sequences from this list. These become your new set of beams.
5.  Repeat until an end-of-sequence token is generated or the maximum length is reached.

**Why it Works:** By keeping multiple hypotheses open, beam search can find a sequence with a higher overall probability, even if it has to choose a slightly less probable word at an early step. It is much better at finding fluent and coherent sequences than greedy search.

**Limitation:** It can still lead to repetitive and somewhat boring text, as it's heavily biased towards high-probability phrases.

---

## Part 3: Controlled Randomness - Advanced Sampling

To make text more creative and human-like, we can introduce randomness by **sampling** from the model's output distribution instead of just taking the `argmax`.

### 3.1. Temperature

*   **The Idea:** We can control the "creativity" or "randomness" of the sampling by adjusting the shape of the probability distribution with a **temperature** parameter.
*   **How it Works:** Before the softmax, we divide the logits by the temperature `T`.
    *   `T > 1`: **Increases randomness**. The distribution becomes flatter, making less likely words more probable. The model becomes more creative and surprising.
    *   `T < 1`: **Decreases randomness**. The distribution becomes sharper, increasing the probability of the most likely words. The model becomes more deterministic and focused.
    *   `T = 1`: Standard softmax.
    *   `T -> 0`: Approaches greedy search.

### 3.2. Top-k Sampling

*   **The Idea:** Instead of considering the entire vocabulary, we restrict the sampling pool to only the `k` most likely next tokens.
*   **How it Works:**
    1.  Get the logits for the next token.
    2.  Identify the top `k` tokens with the highest probabilities.
    3.  Redistribute the probability mass among only these `k` tokens.
    4.  Sample from this new, smaller distribution.
*   **Why it Works:** It prevents bizarre or nonsensical words from being chosen by cutting off the long, low-probability tail of the distribution, while still allowing for some variation among the most likely candidates.

### 3.3. Nucleus (Top-p) Sampling

*   **The Idea:** This is a more dynamic alternative to Top-k. Instead of picking a fixed number `k`, we pick the smallest set of tokens whose cumulative probability is greater than a certain threshold `p`.
*   **How it Works:**
    1.  Get the logits and sort the tokens by their probability in descending order.
    2.  Sum their probabilities from the top until the sum exceeds the threshold `p` (e.g., `p=0.92`).
    3.  The tokens included in this sum form the new sampling pool (the "nucleus").
    4.  Sample from this new distribution.
*   **Why it Works:** The size of the sampling pool is **adaptive**. When the model is very confident about the next word, the nucleus will be very small (maybe only 1 or 2 words). When the model is uncertain, the nucleus will be larger, allowing for more diversity.

**Best Practice:** Top-p (nucleus) sampling is now widely considered the most effective and popular sampling strategy, often giving better results than Top-k.

---

## Part 4: Using the `generate` Method in Hugging Face

The Hugging Face `transformers` library makes it incredibly easy to use all of these techniques through its `.generate()` method.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("--- Part 4: Using the .generate() Method ---")

# --- 1. Load Model and Tokenizer ---
model_name = 'gpt2' # Using the base gpt2 for this demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- 2. Create the Prompt ---
prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley in the Andes Mountains."
inputs = tokenizer(prompt, return_tensors="pt")

# --- 3. Generate with Different Strategies ---

# --- a) Greedy Search (default) ---
print("\n--- a) Greedy Search ---")
greedy_output = model.generate(inputs.input_ids, max_new_tokens=50)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# --- b) Beam Search ---
print("\n--- b) Beam Search (num_beams=5) ---")
beam_output = model.generate(
    inputs.input_ids, 
    max_new_tokens=50, 
    num_beams=5, 
    early_stopping=True
)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# --- c) Top-k and Top-p (Nucleus) Sampling ---
# To enable sampling, we must set do_sample=True
print("\n--- c) Nucleus Sampling (top_p=0.95, top_k=50) ---")
sampling_output = model.generate(
    inputs.input_ids,
    max_new_tokens=50,
    do_sample=True, # Enable sampling
    top_k=50,       # Consider only the top 50 words
    top_p=0.95,     # Consider the smallest set of words whose cumulative prob is > 0.95
    temperature=0.7 # Make the distribution a bit sharper
)
print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))

# --- d) Preventing Repetition ---
print("\n--- d) Sampling with Repetition Penalty ---")
repetitive_output = model.generate(
    inputs.input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    no_repeat_ngram_size=2 # Prevents any 2-gram from being repeated
)
print(tokenizer.decode(repetitive_output[0], skip_special_tokens=True))
```

## Conclusion

High-quality text generation is more than just having a good model; it's about having a smart **decoding strategy**. By moving beyond simple greedy search, we can control the trade-off between coherence and creativity.

**Key Takeaways:**

1.  **Greedy Search is Flawed:** It often leads to repetitive and unnatural text.
2.  **Beam Search Improves Coherence:** By exploring multiple hypotheses, it can find more globally probable sequences, but can still be deterministic and dull.
3.  **Sampling Adds Creativity:** Introducing randomness by sampling from the output distribution is key to generating human-like text.
4.  **Control the Randomness:** Techniques like **Temperature**, **Top-k**, and **Top-p (Nucleus) Sampling** give you fine-grained control over the sampling process. Top-p is often the most effective.
5.  **Use the `.generate()` Method:** The Hugging Face `transformers` library provides a powerful and unified `.generate()` method that implements all of these advanced techniques, making them easy to use and experiment with.

Mastering these generation techniques is essential for any application involving creative text generation, from chatbots and story writing to code completion and summarization.

## Self-Assessment Questions

1.  **Greedy vs. Beam Search:** What is the main difference between greedy search and beam search?
2.  **Temperature:** If you set the temperature to a very high value (e.g., 2.0), how would you expect the generated text to change?
3.  **Top-k vs. Top-p:** What is the main advantage of Top-p (nucleus) sampling over Top-k sampling?
4.  **`no_repeat_ngram_size`:** If you set `no_repeat_ngram_size=3`, what specific behavior are you preventing?
5.  **Hugging Face `generate`:** To use sampling methods like Top-k or Top-p in the `.generate()` method, what boolean parameter must you set to `True`?
