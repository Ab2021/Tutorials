# Day 17.3: GPT-3 & In-Context Learning - A Practical Guide

## Introduction: The Dawn of a New Paradigm

If GPT-2 showed that scale was important, **GPT-3**, released in 2020, was the definitive statement. With **175 billion parameters** (over 100x larger than GPT-2), it was an unprecedented leap in scale. The results were transformative. GPT-3 demonstrated such a strong ability to perform tasks from just a few examples provided in its prompt that it introduced a new paradigm for using large language models: **in-context learning**.

Instead of fine-tuning a model for a specific task, you could now often get impressive results by simply showing the model what you wanted it to do within the prompt itself. This dramatically lowered the barrier to using powerful AI, as no training or gradient updates were required.

This guide explores the key capabilities of GPT-3 and the concept of in-context learning.

**Today's Learning Objectives:**

1.  **Appreciate the Scale of GPT-3:** Understand the leap in model and dataset size from GPT-2 to GPT-3.
2.  **Grasp In-Context Learning:** Learn the difference between zero-shot, one-shot, and few-shot prompting.
3.  **Understand the Shift from Fine-tuning to Prompting:** See why in-context learning has become a dominant way to interact with large language models.
4.  **Explore Prompt Engineering:** Understand the basics of designing effective prompts to guide the model's behavior.
5.  **Use a GPT-style Model for Few-Shot Tasks:** Use the Hugging Face library to simulate few-shot prompting with an open-source GPT-style model.

---

## Part 1: The GPT-3 Model - Scale on an Unprecedented Level

*   **Architecture:** The architecture of GPT-3 is identical to GPT-2—a decoder-only Transformer. The only difference is its massive scale.
*   **Model Size:** The largest version has 96 layers and 175 billion parameters.
*   **Dataset:** Trained on a colossal 570GB dataset called "The Common Crawl," which was filtered to improve quality, along with other high-quality sources like Wikipedia.

This immense scale is what enables its powerful in-context learning abilities.

---

## Part 2: In-Context Learning - Teaching Without Training

**The Core Idea:** A sufficiently large language model develops the ability to recognize patterns and perform tasks described in its prompt, without needing to be fine-tuned. It learns the task "on the fly" from the context you provide.

There are three main levels of in-context learning:

1.  **Zero-Shot Learning:** You ask the model to perform a task without giving it any complete examples. You only provide the instruction and the query.
    *   **Example:** "Translate this English sentence to French: The cat is black."

2.  **One-Shot Learning:** You provide **one** complete example of the task to show the model what you want, followed by your query.
    *   **Example:** "Translate English to French:\nsea otter => loutre de mer\ncheese =>"

3.  **Few-Shot Learning:** You provide a **few** (typically 2 to 5) complete examples before your final query. This is often the most effective approach, as it gives the model a clearer understanding of the desired pattern and output format.
    *   **Example:** "Translate English to French:\nsea otter => loutre de mer\ncheese => fromage\ncar =>"

**Why it Works:** The model, having seen vast amounts of text, has learned to be an extremely good pattern-matcher. When you provide examples in the prompt, it recognizes the pattern (`English => French`) and continues it for your query.

![In-Context Learning](https://i.imgur.com/1fA4s4p.png)

### 2.1. The Shift from Fine-Tuning

| Feature         | Fine-Tuning                                      | In-Context Learning (Prompting)                  |
|-----------------|--------------------------------------------------|--------------------------------------------------|
| **How it works**  | Update the model's weights on a labeled dataset. | No weight updates. Guide the model via its prompt. |
| **Data Needs**    | Requires a (sometimes large) labeled dataset.    | Requires only a few examples for the prompt.     |
| **Computation**   | Requires GPU time for training.                  | Requires only fast, forward-pass inference.      |
| **Flexibility**   | Creates a specialized model for one task.        | Can perform many different tasks with one model. |
| **Performance**   | Often yields the highest possible performance.   | Can be surprisingly effective, sometimes near SOTA. |

---

## Part 3: The Art of Prompt Engineering

With in-context learning, the developer's job shifts from creating datasets and training loops to **designing effective prompts**. This is known as **prompt engineering**.

A good prompt clearly communicates the desired task, format, and context to the model.

**Key Principles of Prompting:**

*   **Be Specific:** Clearly state the instruction. Instead of "Summarize this," try "Summarize this article for a 5th grader in three bullet points."
*   **Provide Good Examples (for Few-Shot):** The examples you provide should be clear, correct, and consistent in format.
*   **Use Delimiters:** Use markers like `###` or `---` to clearly separate instructions, examples, and the final query.
*   **Experiment:** Different phrasings can elicit vastly different responses. Iterating on your prompt design is key.

### 3.1. A Practical Few-Shot Example with Hugging Face

While we can't run the 175B parameter GPT-3 model, we can use a smaller, open-source GPT-style model (like GPT-2 or EleutherAI's GPT-Neo) to demonstrate the principle of few-shot learning.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

print("--- Part 3: Few-Shot Learning in Practice ---")

# --- 1. Load a GPT-style Model and Tokenizer ---
# We use GPT-J-6B, a powerful open-source model, but you can swap it for 'gpt2-xl'
# Note: Running a large model like this requires a powerful GPU and a lot of RAM.
# This code is for demonstration of the concept.

# For a runnable example, change this to 'gpt2'
model_name = 'gpt2' 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- 2. Design the Few-Shot Prompt ---
# The task: Classify a movie review as "Positive" or "Negative".

prompt = """
Review: This movie was incredible, a masterpiece of cinema.
Sentiment: Positive
###
Review: I would not recommend this film to my worst enemy.
Sentiment: Negative
###
Review: A truly heartwarming and beautiful story.
Sentiment: Positive
###
Review: An absolute bore from start to finish.
Sentiment:"""

# --- 3. Generate the Completion ---
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the completion
output_sequences = model.generate(
    input_ids=inputs.input_ids,
    max_new_tokens=1, # We only need one word: Positive or Negative
    temperature=0.1, # Lower temperature makes the output more deterministic
)

# Decode the result
result = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print(f"--- Few-Shot Sentiment Classification ---")
print(f"Prompt:\n{prompt}")
print(f"\nFull Model Output:\n{result}")

# Extract just the final sentiment
final_sentiment = result.split("Sentiment:")[-1].strip()
print(f"\nExtracted Sentiment: {final_sentiment}")
```

## Conclusion: A New Way to Interact with AI

GPT-3 and the paradigm of in-context learning have fundamentally changed how many people develop AI applications. The focus has shifted from the slow and data-intensive process of fine-tuning to the fast and iterative process of prompt engineering.

**Key Takeaways:**

1.  **GPT-3 is About Scale:** Its remarkable abilities are a direct consequence of its massive size and training dataset.
2.  **In-Context Learning is Powerful:** Large language models can perform new tasks just by being shown a few examples in a prompt, with no gradient updates needed.
3.  **Few-Shot is Often Best:** Providing 2-5 examples in the prompt (few-shot) is typically more effective than providing zero or one.
4.  **Prompt Engineering is a Key Skill:** The quality and structure of your prompt are the most important factors in getting good results from a large generative model.

While fine-tuning still provides the highest possible performance for a specific task, in-context learning offers a new, incredibly flexible, and accessible way to harness the power of large language models.

## Self-Assessment Questions

1.  **In-Context Learning:** What is the key difference between fine-tuning and in-context learning?
2.  **Zero-shot vs. Few-shot:** What is the difference between a zero-shot and a few-shot prompt?
3.  **Prompt Engineering:** You want a model to generate a Python function that takes a list of numbers and returns the sum. Write a good few-shot prompt for this task.
4.  **Model Size:** Why is a massive model like GPT-3 able to perform in-context learning while a smaller model like the original GPT-1 cannot?
5.  **Limitations:** What are some potential downsides of relying on in-context learning with a proprietary, API-based model like GPT-3 compared to fine-tuning your own open-source model?
