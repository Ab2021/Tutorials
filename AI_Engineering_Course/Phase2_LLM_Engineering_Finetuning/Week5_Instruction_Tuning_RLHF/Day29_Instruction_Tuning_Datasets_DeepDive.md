# Day 29: Instruction Tuning Datasets
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Self-Instruct Algorithm (Alpaca)

**Goal:** Generate 52k instructions from 175 seed examples.
**Process:**
1. **Seed Pool:** Start with 175 human-written (Instruction, Output) pairs.
2. **Sample:** Randomly select 8 examples from the pool.
3. **Generate:** Prompt GPT-3.5:
   > "Here are 8 instruction-response pairs. Generate a new instruction that is different from these."
4. **Filter:** Check if the new instruction is too similar (ROUGE-L > 0.7) to existing ones. If yes, discard.
5. **Generate Response:** Use GPT-3.5 to generate the output for the new instruction.
6. **Add to Pool:** Add the new pair to the pool.
7. **Repeat:** Until 52k samples.

**Cost:** ~$500 using GPT-3.5-turbo (2023 pricing).

### 2. Evol-Instruct Depth vs. Breadth

**Depth Evolution (In-Depth):**
- Add constraints: "Explain in 3 sentences."
- Add reasoning: "Explain step-by-step."
- Increase difficulty: "Now consider edge cases."

**Breadth Evolution (In-Breadth):**
- Change topic: "Now apply this to biology instead of physics."
- Mutate format: "Convert this to a dialogue."

**Prompt Template:**
```
I want you to act as a Prompt Rewriter.
Your objective is to rewrite the given prompt to make it more complex.
You MUST follow the instructions below:
- Add more constraints/requirements
- Replace general concepts with specific ones
- Increase reasoning steps

#Given Prompt#: {instruction}
#Rewritten Prompt#:
```

### 3. Data Deduplication Pipeline

**Exact Dedup:**
- Hash each instruction. Remove duplicates.

**Near Dedup (MinHash LSH):**
- Convert instruction to n-grams (n=5).
- Hash each n-gram. Keep top-k hashes (MinHash signature).
- Use LSH to find similar signatures.
- If Jaccard similarity > 0.8, remove.

**Semantic Dedup:**
- Embed instructions using Sentence-BERT.
- Cluster embeddings (HDBSCAN).
- Keep only 1 sample per cluster.

### Code: Simple Evol-Instruct

```python
import openai

def evolve_instruction(instruction, evolution_type="depth"):
    if evolution_type == "depth":
        prompt = f"""
Rewrite the following instruction to make it more complex by adding constraints or requiring deeper reasoning.

Original: {instruction}
Evolved:
"""
    else: # breadth
        prompt = f"""
Rewrite the following instruction to cover a different but related topic or format.

Original: {instruction}
Evolved:
"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example
seed = "List 5 programming languages."
evolved = evolve_instruction(seed, "depth")
# Output: "List 5 programming languages, categorize them by paradigm, and explain the primary use case for each."
```
