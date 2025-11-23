# Day 76: Fine-tuning Agents & Function Calling
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why use Grammars (GBNF) instead of just prompting for JSON?

**Answer:**
- **Reliability:** Prompting works 95% of the time. Grammars work 100% of the time (syntactically).
- **Efficiency:** The model doesn't waste tokens generating invalid characters that get rejected.
- **Security:** Prevents injection attacks where the model breaks out of the JSON structure.

#### Q2: What is the difference between "Function Calling" and "Plugins"?

**Answer:**
- **Function Calling:** The capability of the model to output structured tool requests. (The *Mechanism*).
- **Plugins:** The ecosystem/interface wrapping the API. (The *Implementation*).
- OpenAI Plugins use Function Calling under the hood.

#### Q3: How do you handle "Hallucinated Arguments"?

**Answer:**
- **Scenario:** Tool needs `user_id`. Model guesses `123`.
- **Solution:**
  - **Validation:** Check args against DB.
  - **Feedback:** Return error "Invalid user_id" to model.
  - **Fine-tuning:** Train on examples where the model asks "What is your user ID?" instead of guessing.

#### Q4: Explain "NexusRaven" or "Gorilla".

**Answer:**
- These are open-source models specifically fine-tuned for API calling.
- They are trained on massive datasets of API documentation and calls.
- They often beat GPT-4 on retrieving the correct arguments for obscure APIs.

#### Q5: What is the "Context Window" challenge in Tool Use?

**Answer:**
- If you have 100 tools, injecting all their schemas into the system prompt consumes huge context.
- **Solution:** Retrieval-Augmented Tool Use. Retrieve only the top 5 relevant tools based on the user query, then inject those schemas.

---

### Production Challenges

#### Challenge 1: Schema Drift

**Scenario:** API changes `get_weather(city)` to `get_weather(lat, lon)`. Agent breaks.
**Root Cause:** Model trained/prompted on old schema.
**Solution:**
- **Dynamic Prompts:** Fetch tool definitions from code/OpenAPI spec at runtime. Never hardcode schemas in the prompt.

#### Challenge 2: Infinite Tool Loops

**Scenario:** Agent calls `ls`, sees file, calls `ls` again.
**Root Cause:** Model forgets it already ran the tool.
**Solution:**
- **Scratchpad:** Append "I have already run `ls`" to the context.
- **Deduplication:** Prevent calling same tool with same args twice in a row.

#### Challenge 3: Fine-tuning Overfitting

**Scenario:** Fine-tuned agent becomes great at tools but forgets how to chat (Catastrophic Forgetting).
**Root Cause:** Training data was 100% tool calls.
**Solution:**
- **Mix Data:** Include 50% general chat data in the fine-tuning set.

#### Challenge 4: Security (Prompt Injection via Tool Output)

**Scenario:** Tool returns "Delete all files". Model executes it.
**Root Cause:** Trusting tool output.
**Solution:**
- **Sandboxing:** Run tools in Docker.
- **Human Approval:** Require approval for destructive actions.

#### Challenge 5: Latency of Grammar Decoding

**Scenario:** Enforcing grammar slows down generation by 50%.
**Root Cause:** CPU overhead of masking invalid tokens at every step.
**Solution:**
- **Optimized Engines:** Use `xgrammar` or optimized `llama.cpp` builds.
- **Speculative Decoding:** Draft with unconstrained, verify with grammar.

### System Design Scenario: Enterprise API Agent

**Requirement:** Agent that can call 500 internal APIs.
**Design:**
1.  **Registry:** Store OpenAPI specs in Vector DB.
2.  **Retrieval:** Query -> Retrieve top 5 tools.
3.  **Prompt:** Inject top 5 schemas.
4.  **Model:** Fine-tuned Mistral-7B (tuned on internal API logs).
5.  **Constraint:** Use GBNF to ensure valid JSON.
6.  **Execution:** API Gateway.

### Summary Checklist for Production
- [ ] **Data:** Use **Synthetic Data** to train tool use.
- [ ] **Constraint:** Use **Grammars** for 100% valid JSON.
- [ ] **Retrieval:** Retrieve relevant tools if > 10 tools.
- [ ] **Safety:** **Sandbox** tool execution.
- [ ] **Monitoring:** Track **Tool Error Rate**.
