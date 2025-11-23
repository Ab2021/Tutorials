# Day 37: ReAct Pattern & Tool Use
## Core Concepts & Theory

### The ReAct Paradigm

**ReAct = Reasoning + Acting**
Introduced by Yao et al. (2023), ReAct interleaves reasoning traces and task-specific actions.

**Core Insight:**
- Traditional agents either reason (plan) OR act, but not both simultaneously.
- ReAct combines them: reason about what to do, then do it, then reason about the result.

### 1. The ReAct Loop

**Structure:**
```
Thought: <reasoning about current state>
Action: <tool to call>
Observation: <result from tool>
Thought: <reasoning about observation>
Action: <next tool to call>
...
Thought: <final reasoning>
Answer: <final response>
```

**Example:**
```
Question: What is the elevation of the highest peak in the Himalayas?

Thought 1: I need to identify the highest peak in the Himalayas.
Action 1: search["highest peak Himalayas"]
Observation 1: Mount Everest is the highest peak in the Himalayas.

Thought 2: Now I need to find the elevation of Mount Everest.
Action 2: search["Mount Everest elevation"]
Observation 2: Mount Everest has an elevation of 8,849 meters.

Thought 3: I have the answer.
Answer: The elevation of the highest peak in the Himalayas (Mount Everest) is 8,849 meters.
```

### 2. Tool Types

**Information Retrieval:**
- **Search:** Web search, database queries.
- **Lookup:** Wikipedia, knowledge bases.
- **Read:** File reading, document parsing.

**Computation:**
- **Calculator:** Math operations.
- **Code Execution:** Python, JavaScript.
- **Data Analysis:** Pandas, SQL queries.

**External Services:**
- **APIs:** Weather, stocks, maps.
- **Databases:** SQL, NoSQL.
- **Web Scraping:** HTML parsing.

**Creation:**
- **File Writing:** Save results.
- **Image Generation:** DALL-E, Stable Diffusion.
- **Email/Notifications:** Send messages.

### 3. Tool Schema Design

**Good Tool Schema:**
```json
{
  "name": "web_search",
  "description": "Search the web for current information. Use this when you need up-to-date facts, news, or information not in your training data.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query. Be specific and include relevant keywords."
      },
      "num_results": {
        "type": "integer",
        "description": "Number of results to return (1-10)",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

**Key Principles:**
- **Clear Name:** Descriptive, verb-based.
- **Detailed Description:** When to use this tool.
- **Explicit Parameters:** Type, description, constraints.
- **Examples:** Include usage examples in description.

### 4. Prompt Engineering for ReAct

**System Prompt Template:**
```
You are an AI assistant that can use tools to answer questions.

You run in a loop of Thought, Action, Observation.
- Thought: Reason about what to do next
- Action: Call a tool using the format: tool_name[argument]
- Observation: The result will be provided

Available tools:
{tool_descriptions}

Always start with a Thought. End with Answer when you have the final response.

Example:
Thought: I need to find X
Action: search[X]
Observation: <result>
Thought: Now I can answer
Answer: <final answer>
```

**Few-Shot Examples:**
Including 2-3 complete examples in the prompt improves performance by 20-30%.

### 5. Error Handling in ReAct

**Tool Execution Errors:**
```
Action: calculate[10 / 0]
Observation: Error: Division by zero

Thought: The calculation failed. I need to handle this edge case.
Action: finish["Cannot divide by zero"]
```

**No Results:**
```
Action: search["xyzabc123nonexistent"]
Observation: No results found

Thought: The search returned nothing. I should try a different query.
Action: search["alternative query"]
```

**Retry Logic:**
- Max retries: 3
- Exponential backoff for API calls
- Fallback to alternative tools

### 6. ReAct vs. Alternatives

**Chain-of-Thought (CoT):**
- Pure reasoning, no actions.
- **Use Case:** Math problems, logic puzzles.

**ReAct:**
- Reasoning + Actions.
- **Use Case:** Information retrieval, multi-step tasks.

**Plan-and-Execute:**
- Plan all steps upfront, then execute.
- **Use Case:** Well-defined tasks with clear steps.

**Comparison:**
| Approach | Planning | Flexibility | Use Case |
|:---------|:---------|:------------|:---------|
| **CoT** | None | Low | Pure reasoning |
| **ReAct** | Incremental | High | Dynamic tasks |
| **Plan-Execute** | Upfront | Medium | Structured tasks |

### 7. Tool Selection Strategies

**Explicit Selection:**
- Agent explicitly chooses which tool to use.
- **Prompt:** "Which tool should I use?"

**Implicit Selection:**
- Model generates tool call directly.
- **OpenAI Function Calling:** Model outputs JSON function call.

**Routing:**
- Classifier determines which tool is relevant.
- **Benefit:** Faster, cheaper than full LLM reasoning.

### 8. Production Considerations

**Latency:**
- Each tool call adds 1-5 seconds.
- **Mitigation:** Parallel execution, caching.

**Cost:**
- 5-10 LLM calls per task.
- **Mitigation:** Use smaller models for simple reasoning.

**Reliability:**
- Tool failures, network errors.
- **Mitigation:** Retries, fallbacks, timeouts.

**Security:**
- Malicious tool calls, injection attacks.
- **Mitigation:** Input validation, sandboxing, rate limiting.

### Real-World Examples

**Toolformer (Meta, 2023):**
- LLM that learns when and how to use tools.
- **Training:** Self-supervised on tool-augmented text.

**Gorilla (Berkeley, 2023):**
- Fine-tuned LLaMA for API calls.
- **Dataset:** 1600+ API documentation.

**Function Calling (OpenAI, 2023):**
- GPT-3.5/4 with native tool support.
- **Format:** Structured JSON output.

**ChatGPT Plugins:**
- 1000+ community-built tools.
- **Examples:** Wolfram, Zapier, Kayak.

### Summary

**ReAct Benefits:**
- Transparent reasoning (interpretable).
- Flexible (adapts to observations).
- Powerful (combines reasoning + actions).

**ReAct Challenges:**
- Verbose (many tokens).
- Error-prone (tool failures).
- Expensive (multiple LLM calls).

### Next Steps
In the Deep Dive, we will implement a production-grade ReAct agent with error handling, retries, and parallel tool execution.
