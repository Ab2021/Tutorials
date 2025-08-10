# Day 17.2: GPT-2 & Scaling Laws - A Practical Guide

## Introduction: The Power of Scale

GPT-1 was a proof of concept, demonstrating that a decoder-only Transformer could be effectively pre-trained for generative tasks. **GPT-2**, released by OpenAI in 2019, was a dramatic demonstration of what happens when you **scale** this idea up. It was, in essence, a much larger version of GPT-1 trained on a massive, high-quality dataset.

The results were stunning. GPT-2 could generate long, coherent, and often contextually relevant paragraphs of text from a simple prompt, without any task-specific fine-tuning. This release highlighted the importance of **scaling laws** in deep learning: the observation that performance on many tasks improves predictably as you increase model size, dataset size, and the amount of compute used for training.

This guide explores the key aspects of GPT-2 and the scaling laws that it helped to popularize.

**Today's Learning Objectives:**

1.  **Understand the GPT-2 Architecture:** See how it is a direct, but larger, successor to GPT-1.
2.  **Learn about the WebText Dataset:** Appreciate the importance of a large, high-quality dataset for pre-training.
3.  **Grasp the Concept of Zero-Shot Learning:** Understand how a sufficiently large language model can perform tasks it was never explicitly trained on, simply by being prompted correctly.
4.  **Explore the Scaling Laws:** Learn the relationship between model performance, model size (number of parameters), dataset size, and compute.
5.  **Use a Pre-trained GPT-2 Model:** Use the Hugging Face `transformers` library to perform zero-shot tasks and text generation with a real GPT-2 model.

---

## Part 1: The GPT-2 Model - Bigger is Better

There were very few architectural changes between GPT-1 and GPT-2. The primary difference was scale.

*   **Architecture:** Still a decoder-only Transformer. Minor changes included moving the layer normalization to the input of each sub-block and a modified initialization scheme.
*   **Model Sizes:** GPT-2 was released in several sizes, with the largest being a 48-layer Transformer with 1.5 billion parameters (compared to GPT-1's 117 million).
*   **Dataset (WebText):** This was a key innovation. Instead of using existing datasets, OpenAI created a new, massive (40GB) dataset by scraping text from outbound Reddit links that were highly upvoted. This acted as a quality filter, resulting in a cleaner and more coherent training corpus than just scraping the whole web.

**The Result:** A model that was exceptionally good at its core task of next-word prediction. It was so good, in fact, that it learned to perform other tasks without being explicitly trained on them.

---

## Part 2: Zero-Shot Task Performance

This was the most surprising finding from the GPT-2 paper. Because the model was trained to predict the next word on such a vast and diverse dataset, it implicitly learned to perform tasks like translation, summarization, and question answering.

**The Idea of Zero-Shot Learning:**
You can prompt the model to perform a task by framing it as a text completion problem.

*   **For Translation:**
    *   Prompt: "translate to french: sea otter => loutre de mer, cheese =>"
    *   Model's likely completion: `fromage`

*   **For Summarization:**
    *   Prompt: "My second grader asked me what this passage means: ... [long passage] ... I told him: ..."
    *   Model's likely completion: A simplified summary of the passage.

*   **For Question Answering:**
    *   Prompt: "Q: Who was president of the United States in 1955? A:"
    *   Model's likely completion: `Dwight D. Eisenhower`

The model was never explicitly trained on `(question, answer)` pairs, but it learned these relationships from the patterns in the WebText data.

### 2.1. Performing Zero-Shot Tasks with Hugging Face

Let's use a pre-trained GPT-2 model to perform a simple zero-shot summarization task.

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("---" + " Part 2: Zero-Shot Task Performance with GPT-2" + " ---")

# --- 1. Load Model and Tokenizer ---
# We use the medium-sized GPT-2 model (355M parameters)
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# --- 2. Create the Zero-Shot Prompt ---
article = """
Scientists have discovered a new species of deep-sea fish, which they have named the Stargazer Brooder. 
This remarkable creature is notable for its unique reproductive strategy. The female carries her fertilized eggs 
in her mouth until they hatch, a behavior previously unknown in this family of fish. This ensures the 
young are protected from predators during their most vulnerable stage. The Stargazer Brooder lives at 
depths of over 2,000 meters, in complete darkness, where it uses bioluminescent lures to attract prey.
"""

# We frame the task as a completion problem.
prompt = f"{article}\n\nTl;dr:"

# --- 3. Generate the Completion ---
inputs = tokenizer(prompt, return_tensors="pt")

# Use the .generate() method
summary_ids = model.generate(
    inputs.input_ids, 
    max_new_tokens=25, # Generate a short summary
    num_beams=5, # Beam search can produce better results
    no_repeat_ngram_size=2, # Prevent repetitive phrases
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Original Article (first 50 chars): {article[:50]}...")
print(f"\nGenerated Summary (from prompt '{prompt[-8:]}'):")
# We can post-process the output to just get the summary part.
summary_text = summary.split("Tl;dr:")[1].strip()
print(summary_text)
```

---

## Part 3: The Scaling Laws

GPT-2's success was a key piece of evidence for the **scaling laws** for language models. Researchers at OpenAI performed a systematic study and found that model performance (measured by the test loss) improves predictably as you increase three factors:

1.  **Model Size (N):** The number of parameters in the model.
2.  **Dataset Size (D):** The number of tokens in the training dataset.
3.  **Compute (C):** The amount of computing power used for training.

They found that the test loss follows a **power law** with respect to each of these factors. This means that if you plot the loss against any of these factors on a **log-log plot**, you get a straight line.

![Scaling Laws](https://i.imgur.com/2qZ3V8k.png)

**The Implications:**
*   **Predictability:** This is a huge deal. It means you can train a small model for a short time to estimate the slope of the power law, and then accurately predict how much better your model will get if you spend 10x more compute or collect 10x more data. This allows for much more strategic research and investment.
*   **Bigger is (Often) Better:** The scaling laws suggest that we haven't yet hit the point of diminishing returns. Simply scaling up existing architectures continues to yield better models.
*   **Compute is Key:** The main bottleneck to better performance is often the amount of available compute.

**The Optimal Allocation:**
The studies also showed that for a given compute budget, it's best not to over-invest in model size. The best performance is achieved when you scale up both the model size and the dataset size together according to a specific ratio.

---

## Conclusion: The Era of Scale

GPT-2 marked a turning point in NLP. It shifted the focus from complex, novel architectures to the power of **scale**. It showed that a relatively simple architecture, when trained on a massive, clean dataset with enough compute, could learn to perform a surprising range of tasks without any explicit supervision.

**Key Takeaways:**

1.  **Scale is a Key Ingredient:** GPT-2's primary innovation over GPT-1 was its scale (more parameters, more data).
2.  **Zero-Shot Learning Emerges:** Large language models can perform tasks they weren't explicitly trained for by framing the task as a text completion problem.
3.  **High-Quality Data is Crucial:** The use of the filtered WebText dataset was a key contributor to GPT-2's success and coherence.
4.  **Scaling Laws Provide a Roadmap:** Performance in language models improves as a predictable power law of model size, dataset size, and compute. This insight has guided the development of all subsequent large language models.

GPT-2 and the scaling laws it helped to uncover set the stage for the next generation of even larger and more capable models, like GPT-3, which would take these principles to their logical conclusion.

## Self-Assessment Questions

1.  **GPT-1 vs. GPT-2:** What were the two main differences between the GPT-1 and GPT-2 projects?
2.  **Zero-Shot Learning:** How would you phrase a prompt to a model like GPT-2 to try and get it to answer a question?
3.  **WebText:** What was the source of the WebText dataset, and why was this a clever choice?
4.  **Scaling Laws:** According to the scaling laws, if you plot the test loss versus the number of model parameters on a log-log plot, what shape would you expect the graph to have?
5.  **Practical Implications:** You are a researcher with a fixed compute budget. The scaling laws suggest that to get the best possible model, should you spend your entire budget on making the model as large as possible, or is there a better strategy?

