# Day 76: Fine-tuning Agents & Function Calling
## Core Concepts & Theory

### Why Fine-tune Agents?

**General Models (GPT-4):** Good at everything, master of none.
**Fine-tuned Agents:**
- **Better Tool Use:** Learn specific JSON schemas perfectly.
- **Better Planning:** Learn enterprise-specific workflows.
- **Cheaper:** A fine-tuned Llama-3-8B can beat GPT-4 on specific tasks.
- **Faster:** Less prompting required (system prompt baked into weights).

### 1. Function Calling (Tool Use)

**Concept:**
- LLM outputs structured data (JSON) instead of text.
- **Process:**
  1.  User: "What's the weather?"
  2.  LLM: `call_tool("get_weather", {"city": "NY"})`
  3.  System: Executes tool. Returns "20C".
  4.  LLM: "It is 20C in NY."

**Training:**
- Train on `(User Prompt, Tool Definition, Tool Call)` triples.
- Special tokens: `<tool_start>`, `<tool_end>`.

### 2. Agent Tuning Datasets

**Sources:**
- **Glaive:** Synthetic data for tool use.
- **ToolBench:** Large-scale instruction tuning for tools.
- **Self-Play:** Have GPT-4 generate trajectories, filter for success, train Llama-3 on them.

### 3. Format Enforcement

**Problem:** LLM generates invalid JSON.
**Solution:**
- **Grammar-Constrained Decoding (GBNF):** Force the token generation to follow a grammar (e.g., valid JSON).
- **Libraries:** `llama.cpp` grammars, `outlines`.

### 4. LoRA for Agents

**Strategy:**
- Keep base model frozen.
- Train LoRA adapter on agent trajectories.
- **Benefit:** Can have different adapters for different roles (Coder Adapter, Writer Adapter).

### 5. Berry-picking (Iterative Refinement)

**Concept:**
- Fine-tune the model not just to output the final answer, but to ask clarifying questions.
- **Training Data:** Examples where the agent asks for missing info instead of hallucinating.

### 6. Evaluation of Tool Use

**Metrics:**
- **Syntax Error Rate:** How often is the JSON invalid?
- **Hallucination Rate:** Calling non-existent tools.
- **Argument Accuracy:** Passing correct parameters.

### 7. FireFunction / NexusRaven

**Models:**
- Open-source models specifically fine-tuned for function calling.
- Often outperform larger general models on tool use benchmarks.

### 8. Summary

**Fine-tuning Strategy:**
1.  **Data:** Generate synthetic trajectories using GPT-4.
2.  **Format:** Use standard **OpenAI Function Calling** format.
3.  **Model:** Fine-tune **Llama-3-8B** or **Mistral**.
4.  **Constraint:** Use **Grammars** during inference.
5.  **Eval:** Test on **ToolBench**.

### Next Steps
In the Deep Dive, we will implement a Function Calling Fine-tuning loop, a Grammar-constrained generation script, and a Synthetic Data generator.
