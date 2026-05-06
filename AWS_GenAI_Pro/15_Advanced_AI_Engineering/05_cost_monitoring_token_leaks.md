# 💸 Cost Monitoring & Hidden Token Leaks (Deep Dive)

In Generative AI, code efficiency directly equates to financial efficiency. Unoptimized LLM architectures suffer from **Hidden Token Leaks**, where exponential cost growth occurs due to redundant data transmission.

## 1. The Economics of Token Consumption

Models charge differently for input (prefill) and output (generation) tokens.
- **Claude 3.5 Sonnet:** $3.00 / 1M Input Tokens | $15.00 / 1M Output Tokens.
- Output tokens are 5x more expensive. Any leak involving output tokens (like verbose agent thinking) is financially devastating.

---

## 2. Anatomy of the 3 Major Token Leaks

### A. The Unbounded Chat History Leak
LLMs are stateless. To maintain a conversational memory, developers must pass the entire chat history back to the LLM on every turn.
- Turn 1: 100 tokens.
- Turn 15: 8,000 tokens (Turn 1 through 14 appended together).
This creates an $O(N^2)$ cost curve per session. A user chatting for 30 minutes can burn dollars of API credits.

**Mitigation: Sliding Window Memory (LangChain Implementation)**
```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_aws import ChatBedrock

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")

# Caps the memory at 2000 tokens. 
# Once exceeded, the oldest messages are dropped from the context window.
memory = ConversationTokenBufferMemory(
    llm=llm, 
    max_token_limit=2000, 
    return_messages=True
)
```
*Advanced variant:* **Summarization Memory**. When the token limit is hit, an LLM summarizes the dropped messages into a 150-token paragraph.

### B. Tool Schema Bloat in Agentic Workflows
When using ReAct agents, you must pass the OpenAPI JSON schemas of available tools in the system prompt.
- If an agent has 50 tools, the JSON schemas might consume 20,000 input tokens.
- In a ReAct loop, the agent might take 6 steps to solve a problem. That's `20,000 * 6 = 120,000` input tokens for a single user request.

**Mitigation: Dynamic Tool Retrieval (RAG for Tools)**
Do not inject all 50 schemas. Embed the tool descriptions. When the user asks "Book a flight", run a semantic search against the tool embeddings, retrieve only the `book_flight` and `check_weather` schemas, and inject those 2 tools (800 tokens) into the prompt.

### C. Agentic "Thought" Bloat
In ReAct, the LLM outputs a `Thought` before an `Action`. 
`Thought: Let me analyze the data to see if I need to call the database again...`
Because Output Tokens are 5x more expensive, overly verbose thoughts drain budgets rapidly.
**Mitigation:** Enforce strict brevity in the system prompt: `CRITICAL: Your 'Thought' field must be less than 15 words.`

---

## 3. AWS Cost Governance & Monitoring Architecture

To detect token leaks before they hit the monthly bill, you must instrument your AWS architecture.

### A. Model Invocation Logging to S3
You cannot manage costs using standard CloudWatch application logs because they don't capture native token metrics. You must enable **Bedrock Model Invocation Logging**.
- Routes massive JSON logs containing `inputTokenCount`, `outputTokenCount`, `latency`, and `modelId` directly to S3.

### B. Cost Allocation Tags
Tag every API call.
```python
response = bedrock.invoke_model(
    modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=body,
    tags=[
        {'key': 'CostCenter', 'value': 'HR-Bot'},
        {'key': 'Environment', 'value': 'Production'}
    ]
)
```

### C. Amazon Athena Query for Leak Detection
Once logs are in S3, use Amazon Athena to detect sessions where token usage is growing exponentially (indicating a chat history leak).
```sql
SELECT 
    json_extract_scalar(requestMetadata, '$.sessionId') AS session_id,
    SUM(CAST(json_extract_scalar(tokenUsage, '$.inputTokens') AS INT)) AS total_input_tokens,
    SUM(CAST(json_extract_scalar(tokenUsage, '$.outputTokens') AS INT)) AS total_output_tokens
FROM bedrock_invocation_logs
GROUP BY session_id
HAVING total_input_tokens > 50000 -- Flags runaway sessions
ORDER BY total_input_tokens DESC;
```

### D. AWS Cost Anomaly Detection
Configure AWS Budgets with Anomaly Detection specifically filtered to the `Amazon Bedrock` service. This uses Machine Learning to detect sudden spikes in spend (e.g., a dev accidentally pushing a loop to production) and sends an SNS alert within hours.

---

## 4. Exam & Interview Practice Questions

**Q1: A development team deployed an autonomous customer service agent. Over the weekend, a single user engaged the bot in a 150-turn conversation. The AWS bill spiked by $400 from this single session. The team needs to prevent this from happening again without completely destroying the bot's ability to answer multi-part questions spanning recent context. What is the most cost-effective architectural fix?**
- A) Set up an AWS WAF rate limit restricting users to 5 messages per hour.
- B) Implement a Sliding Window memory architecture (like ConversationTokenBufferMemory) that only retains the last 2000 tokens of conversational turns in the context window.
- C) Enable Bedrock Provider-Level Prompt Caching to cache the chat history.
- D) Switch the model to Amazon Titan Embeddings to reduce token costs.
**Answer: B.** A sliding window memory caps the maximum input tokens sent to the LLM, flattening the exponential $O(N^2)$ cost curve of long chat sessions while preserving recent conversational context. WAF (A) destroys UX. Prompt Caching (C) doesn't stop the leak, it just slightly discounts it. Embeddings (D) don't generate text.

**Q2: An enterprise agent has access to 200 backend APIs, documented via OpenAPI JSON schemas. Currently, all 200 schemas are injected into the agent's system prompt, causing the input token count to hit 80,000 tokens per invocation. How can the architect reduce this hidden token leak while still allowing the agent access to all APIs?**
- A) Use Semantic Caching to store the API responses.
- B) Implement Dynamic Tool Retrieval by using a vector search to inject only the schemas of the top 5 most relevant tools into the prompt based on the user's query.
- C) Compress the JSON schemas into CSV format to save space.
- D) Quantize the LLM to INT4 to handle larger context windows.
**Answer: B.** Dynamic Tool Retrieval (or RAG for Tools) ensures the LLM only reads the specific tool instructions it needs for that specific turn, drastically reducing input tokens from 80k to perhaps 2k.

**Q3: A company wants granular visibility into exactly how many input and output tokens are being consumed by their "HR-Assistant" vs their "IT-Assistant", both of which use Claude 3 Sonnet on Amazon Bedrock. What is the AWS Well-Architected way to achieve this?**
- A) Parse CloudTrail logs for the `InvokeModel` API call.
- B) Enable Bedrock Model Invocation Logging to S3, apply Cost Allocation Tags to the requests, and query the usage with Amazon Athena.
- C) Create a Lambda function to intercept all Bedrock calls and write the payload size to DynamoDB.
- D) Use Amazon Macie to inspect the token counts.
**Answer: B.** Model Invocation Logging is the native Bedrock feature designed specifically to capture granular token usage, latency, and metadata for cost attribution and analytics via Athena. CloudTrail does not log payload token counts.
