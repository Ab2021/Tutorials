# EVALUATION FRAMEWORKS & QUALITY ASSURANCE (v1 - CONCEPTUAL)
## How to prove an AI system works before deploying it (No Code)

---

## 1. THE EVALUATION CRISIS IN AI

Traditional software engineering relies on Unit Tests: given Input X, assert Output Y. This works because traditional functions are deterministic. 

LLMs are non-deterministic. If a client asks, "Summarize this contract," the LLM will generate a different summary every time. You cannot write a unit test for this. 
If an AI Engineer tells an SME client, "I tested it manually and it looks good," the client will not sign the contract. You must architect a programmatic evaluation pipeline.

---

## 2. THE RAGAS FRAMEWORK: DECONSTRUCTING QUALITY

To evaluate a RAG system, you must break "quality" down into independent mathematical metrics. Ragas uses an "LLM-as-a-Judge" architecture. It uses a highly capable model (like GPT-4o) to grade the outputs of your actual system.

### Metric 1: Faithfulness (The Anti-Hallucination Metric)
-   **The Question:** Did the AI make anything up?
-   **The Logic:** The Judge LLM reads the generated answer and extracts every single factual claim. Then, it cross-references each claim against the retrieved context documents. 
-   **The Score:** (Number of supported claims) / (Total claims). 
-   **Architectural Fix:** If Faithfulness is low, the LLM is ignoring the context and using its training data. You must lower the temperature and rewrite the system prompt to aggressively enforce "Answer ONLY from context."

### Metric 2: Answer Relevancy
-   **The Question:** Did the AI actually answer the user's question, or did it dodge it?
-   **The Logic:** The Judge LLM reads the generated answer and tries to reverse-engineer what the original question was. It then calculates the vector similarity between its reverse-engineered question and the user's actual question.
-   **Architectural Fix:** If Relevancy is low, the prompt is likely too vague, causing the LLM to give generic summaries rather than direct answers.

### Metric 3: Context Precision (Noise Reduction)
-   **The Question:** Did we retrieve the right documents, and were they at the top of the list?
-   **The Logic:** Checks the retrieved chunks. If chunk #1 is irrelevant but chunk #5 contains the answer, Precision is penalized.
-   **Architectural Fix:** If Precision is low, you are retrieving too much garbage. You must implement a Cross-Encoder Reranker to push the truly relevant chunks to the top.

### Metric 4: Context Recall (Information Capture)
-   **The Question:** Did we retrieve everything needed to answer the question?
-   **The Logic:** Requires a "Golden Dataset" with known correct answers. It checks if the retrieved chunks contain all the necessary facts to form the correct answer.
-   **Architectural Fix:** If Recall is low, your vector search failed. You must adjust your chunking strategy, switch to Hybrid Search, or use a better embedding model.

---

## 3. PROMPTFOO: REGRESSION TESTING FOR PROMPTS

When you change a system prompt to fix an edge case, how do you know you didn't break 50 other things? This is Prompt Regression.

### The CI/CD Pipeline for Prompts
-   **The Tool:** Promptfoo allows you to treat prompts like code. 
-   **The Matrix:** You define a grid. On the X-axis: your prompt variations. On the Y-axis: 100 test cases (user inputs). On the Z-axis: 3 different LLMs (GPT-4o, Claude 3.5, Llama 3).
-   **The Assertions:** Instead of exact string matching, you write assertions like `contains-json`, `latency < 5000ms`, or use LLM-based rubrics (`assert that the tone is professional`).
-   **The Execution:** When a developer commits a prompt change, Promptfoo runs the matrix in the CI/CD pipeline. If the new prompt causes the system to fail previously passing tests, the deployment is blocked.

---

## 4. BUILDING THE GOLDEN DATASET

The hardest part of AI evaluation in consulting is getting the ground truth. An SME client does not have a dataset of 1,000 perfectly annotated questions and answers.

### The "Shadow Mode" Architecture
1.  **Phase 1 (Collection):** Deploy a simple baseline RAG system internally to a few domain experts at the client company. Log every single question they ask and the documents retrieved.
2.  **Phase 2 (Synthetic Generation):** Pass the client's documents through an LLM instructed to "Generate 100 realistic questions a user might ask based on this document, and provide the correct answer."
3.  **Phase 3 (Human Review):** Present this synthetic dataset to the client's domain experts. Have them correct the answers. This becomes your Golden Dataset.

---

## 5. INTERVIEW Q&A DRILL-DOWN: EVALUATION

**Q: A client says the AI is giving "bad answers." How do you debug this?**
**Strategy:** Implement the 4-Level Failure Taxonomy.
**Answer:** "When an AI fails, I do not just tweak the prompt. I run a structured root-cause analysis based on four levels. 
Level 1 (Coverage): Does the document containing the answer actually exist in our database? 
Level 2 (Retrieval): If it exists, did our vector search actually retrieve it in the top 5 chunks? 
Level 3 (Context Quality): If it was retrieved, was the chunk truncated or missing critical surrounding context? 
Level 4 (Generation): If perfect context was provided, did the LLM hallucinate or ignore it? 
I isolate the failure point. If it's a retrieval failure, I fix the embedding or chunking strategy. If it's a generation failure, I fix the prompt. Guessing wastes time; telemetry solves the problem."

**Q: You deploy an AI system. A week later, OpenAI releases a new model, and you want to upgrade. How do you ensure it is safe?**
**Strategy:** Emphasize Regression Testing.
**Answer:** "I never swap models in production blindly, even for a supposedly 'better' model. New models have different latent behaviors. I would use a framework like Promptfoo to run our Golden Dataset of 200 historically verified queries against the new model in a staging environment. I run Ragas metrics to check if Faithfulness or Context Recall dropped. Only if the new model achieves statistical parity or improvement across all assertions in the automated test suite do I authorize the production swap."

**Q: In an Agentic system, how do you evaluate if the Agent is performing well, since it takes multiple unpredictable steps?**
**Strategy:** Shift from Output Evaluation to Trajectory Evaluation.
**Answer:** "Evaluating agents requires Trajectory Evaluation. I don't just look at the final answer; I evaluate the path the agent took. I define golden test cases with 'Expected Tool Sequences'. For example, if the task is 'Reorder low stock', the expected trajectory is [Check_Inventory -> Get_Price -> Draft_Order -> Request_Human_Approval]. If the agent completes the task but skips the 'Request_Human_Approval' tool, that is a catastrophic trajectory failure, even if the final output looks correct. I build evaluation suites that penalize the agent for taking unnecessary steps or skipping mandatory guardrail tools."
