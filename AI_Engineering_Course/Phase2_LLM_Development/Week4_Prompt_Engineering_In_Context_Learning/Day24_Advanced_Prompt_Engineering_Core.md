# Day 24: Advanced Prompt Engineering Techniques
## Core Concepts & Theory

### Beyond "Just Asking"

Standard prompting ("Zero-Shot") often fails on complex reasoning tasks.
**Why?** LLMs are autoregressive. They generate the answer token-by-token. If the reasoning requires multiple steps, the model needs "scratchpad" space to compute intermediate results before outputting the final answer.

### 1. Chain-of-Thought (CoT)

**Concept:** Explicitly encourage the model to generate a series of intermediate reasoning steps.
**Zero-Shot CoT:** Append "Let's think step by step." to the prompt. (Kojima et al., 2022).
**Few-Shot CoT:** Provide examples of (Question, Reasoning, Answer).
- **Impact:** Massive performance boost on Math (GSM8K) and Logic tasks.

### 2. Self-Consistency

**Concept:** LLMs are probabilistic. A single generation might be a fluke.
**Algorithm:**
1.  Generate $N$ different reasoning paths (using high temperature, e.g., 0.7).
2.  Extract the final answer from each path.
3.  **Majority Vote:** Select the most common answer.
- **Benefit:** Significantly improves reliability over greedy decoding.

### 3. Tree of Thoughts (ToT)

**Concept:** Generalize CoT from a linear path to a tree search.
**Algorithm:**
1.  **Decomposition:** Break problem into steps.
2.  **Generation:** Generate multiple candidates for the next step.
3.  **Evaluation:** Self-evaluate each candidate (Is this promising?).
4.  **Search:** Use BFS or DFS to explore the tree of possibilities.
- **Use Case:** Creative writing, Crosswords, complex planning.

### 4. ReAct (Reasoning + Acting)

**Concept:** Interleave reasoning traces with action execution.
**Loop:**
1.  **Thought:** "I need to find the age of Obama."
2.  **Action:** `Search("Obama age")`
3.  **Observation:** "Barack Obama is 62 years old."
4.  **Thought:** "Now I can answer."
5.  **Answer:** "He is 62."
- **Impact:** Enables LLMs to use tools and interact with the world.

### 5. RAG (Retrieval-Augmented Generation)

**Concept:** Provide the model with external knowledge.
**Process:**
1.  **Retrieve:** Find relevant documents from a vector DB.
2.  **Augment:** Paste documents into the context window.
3.  **Generate:** "Answer the question based ONLY on the context below..."

### Summary of Techniques

| Technique | Mechanism | Best For |
| :--- | :--- | :--- |
| **Zero-Shot** | Direct Answer | Simple tasks |
| **CoT** | "Think step by step" | Math, Logic |
| **Self-Consistency** | Majority Vote | Reliability |
| **ToT** | Tree Search | Planning, Puzzles |
| **ReAct** | Thought-Action Loop | Tool Use, Agents |

### Next Steps
In the Deep Dive, we will implement a Tree of Thoughts solver for a logic puzzle using OpenAI's API.
