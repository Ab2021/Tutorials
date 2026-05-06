# 40 Practical Debugging Questions – AIP-C01
## Real-World Failure Scenarios & Root-Cause Analysis

> Each question describes a BROKEN system. Find the root cause and fix.

---

### SECTION A — RAG System Failures (Q1–Q12)

**Q1.** A RAG chatbot for a bank returns accurate responses for English queries but completely misses documents when users ask in Spanish, despite the knowledge base containing both English and Spanish documents.

What is the ROOT CAUSE?

- A. The knowledge base doesn't support multilingual documents  
- B. The embedding model used during ingestion does not produce language-agnostic embeddings — Spanish queries and English documents land in different vector spaces, causing retrieval failure  
- C. Bedrock Guardrails is blocking Spanish inputs  
- D. The chunk size is too large for Spanish text  

**Answer: B**  
*Debug: Verify the embedding model supports multilingual embeddings (Cohere Embed Multilingual or Amazon Titan with multilingual support). The SAME model must be used for both ingestion AND query-time embedding.*

---

**Q2.** A legal RAG system retrieves the correct documents every time, but responses contain information NOT present in any retrieved document.

What is HAPPENING and how do you fix it?

- A. The retrieval step is broken — it's retrieving the wrong documents  
- B. The FM is hallucinating — adding information beyond the retrieved context. Fix: Enable Bedrock Guardrails grounding check with a high threshold (e.g., 0.8)  
- C. The chunk size is too small  
- D. The metadata filters are excluding relevant documents  

**Answer: B**  
*Debug: Faithfulness metric will show low score. The model is not "sticking to context." High grounding threshold blocks responses not supported by retrieved documents.*

---

**Q3.** After increasing chunk size from 300 to 1,500 tokens to "improve context," a developer notices retrieval precision dropped sharply and users get less relevant answers.

Why did this happen?

- A. Larger chunks use more memory and slow down the server  
- B. Larger chunks cause semantic dilution — the embedding vector represents a blend of multiple topics, making it harder to match specific queries precisely  
- C. The vector store cannot handle chunks larger than 512 tokens  
- D. The embedding model truncates chunks over 1,000 tokens  

**Answer: B**  
*Debug: Reduce chunk size. Consider hierarchical chunking — small child chunks for retrieval, larger parent chunks for context delivery to the model.*

---

**Q4.** A document ingestion pipeline successfully syncs files to the Bedrock Knowledge Base, but the pipeline runs every 5 minutes using EventBridge, even when no files have changed. This causes unnecessary API costs.

What is the better architectural pattern?

- A. Increase the EventBridge interval to 60 minutes  
- B. Replace EventBridge polling with S3 Event Notifications → Lambda → StartIngestionJob (event-driven — only triggers when files actually change)  
- C. Add a Lambda function that checks if files changed before calling StartIngestionJob  
- D. Use S3 Intelligent-Tiering to reduce costs  

**Answer: B**  
*Debug: EventBridge scheduled polling wastes calls when nothing has changed. S3 event-driven architecture only fires on actual object creation/modification events.*

---

**Q5.** A RAG system that worked correctly yesterday is now returning responses based on outdated company policies, even though new policy documents were uploaded to S3.

What should be checked FIRST?

- A. Check if the foundation model was updated  
- B. Verify the ingestion job ran successfully after the S3 upload (check StartIngestionJob status and completion)  
- C. Check if the vector store is full  
- D. Verify the embedding model is still available  

**Answer: B**  
*Debug: Uploading to S3 does NOT automatically update the Knowledge Base. The ingestion job must complete successfully. Check job status in Bedrock console or via API — it may have failed silently.*

---

**Q6.** A medical RAG application's retrieval returns 10 chunks per query, but developers notice the same document appears 6–7 times in each result set, crowding out other relevant sources.

What causes this and how do you fix it?

- A. The knowledge base has duplicate documents that need to be removed  
- B. The chunking strategy produces many similar child chunks from a single document, which all score high for similar queries. Fix: Enable maximum marginal relevance (MMR) or diversity-based re-ranking to ensure result diversity  
- C. The OpenSearch index is corrupted  
- D. The embedding model is biased toward this document type  

**Answer: B**  
*Debug: Enable result diversity / MMR reranking in retrieval configuration. Alternatively, limit results per document source using metadata-based post-filtering.*

---

**Q7.** A RAG evaluation job reports high faithfulness (0.92) but very low correctness (0.31). What does this mean?

- A. The model is hallucinating frequently  
- B. The model is faithfully citing retrieved content, but the knowledge base itself contains outdated or incorrect information. The problem is DATA QUALITY, not the model  
- C. The evaluation job is misconfigured  
- D. The model needs fine-tuning  

**Answer: B**  
*Debug: Faithfulness = model sticks to what it retrieved. Correctness = retrieved content matches ground truth. Gap = KB has stale/wrong data. Solution: Update and re-ingest the knowledge base with correct documents.*

---

**Q8.** After migrating a knowledge base from fixed-size chunking to semantic chunking, ingestion time increased 10x and costs spiked significantly.

What is the cause and is this expected?

- A. Semantic chunking has a bug that re-processes documents multiple times  
- B. Semantic chunking requires an LLM call to identify semantic boundaries, making it significantly more expensive and slower than fixed-size chunking. This is expected — trade cost for retrieval quality  
- C. The vector store needs to be scaled up  
- D. The embedding model is incompatible with semantic chunks  

**Answer: B**  
*Debug: Semantic chunking is inherently more expensive. Evaluate if the retrieval quality improvement justifies the cost. For cost-sensitive workloads, hierarchical chunking may be a better balance.*

---

**Q9.** A developer implements RAG regression testing using SHA-256 hashes of responses. After each knowledge base update, 100% of hashes differ, generating alerts even when answers remain correct.

What is the fundamental flaw?

- A. SHA-256 is not supported for text hashing  
- B. LLMs are non-deterministic — temperature > 0 causes probabilistic token sampling, so even identical correct answers produce different wording and thus different hashes every run  
- C. The test dataset is too small  
- D. The knowledge base is being cleared between runs  

**Answer: B**  
*Fix: Replace response hashing with Bedrock Model Evaluation jobs using Correctness and Completeness metrics. These handle paraphrasing and semantic equivalence.*

---

**Q10.** A knowledge base retrieves documents correctly for short queries (5–10 words) but fails for detailed multi-sentence queries.

What is the likely cause?

- A. Bedrock has a query length limit  
- B. Long queries exceed the embedding model's token limit — the query gets truncated, losing key semantic information. The truncated embedding no longer matches the relevant document chunks  
- C. The OpenSearch index cannot handle complex queries  
- D. Metadata filters are blocking results for long queries  

**Answer: B**  
*Debug: Check the embedding model's maximum input token limit. For long queries, consider query compression/summarization before embedding, or use an embedding model with a longer context window.*

---

**Q11.** Retrieval scores are consistently high (>0.85) but the generated answers are completely wrong for complex multi-hop questions requiring information from multiple documents.

What is the issue?

- A. The vector store is returning wrong documents  
- B. The retrieval step finds individual relevant chunks but the chunks are isolated — they lack the connective context needed for multi-hop reasoning. The model can't "connect the dots" across separate chunks  
- C. The foundation model doesn't support multi-hop reasoning  
- D. The temperature setting is too high  

**Answer: B**  
*Fix: Implement hierarchical chunking to return larger parent chunks with more context. Or implement a graph-based RAG approach (Neptune Analytics) that explicitly models document relationships. Alternatively, increase the number of retrieved chunks.*

---

**Q12.** A team enables metadata filtering on their knowledge base using `{"equals": {"key": "status", "value": "active"}}`, but retrieval returns zero results even though active documents exist.

What is wrong?

- A. OpenSearch Serverless doesn't support equality filters  
- B. The metadata field "status" was not set during ingestion — documents were indexed without the metadata tag, so the filter finds no matching documents  
- C. The filter syntax is incorrect  
- D. Active documents are in a different knowledge base  

**Answer: B**  
*Debug: Check document metadata during ingestion. Metadata fields must be explicitly set on documents when added to the knowledge base. Re-ingest documents with proper metadata tags.*

---

### SECTION B — Agent & Orchestration Failures (Q13–Q22)

**Q13.** A Bedrock Agent designed to look up customer orders keeps asking the user for their order ID repeatedly, even after the user provided it clearly in the first message.

What should the developer investigate?

- A. Increase the agent's Lambda function timeout  
- B. Enable Agent Traces and examine the observation returned from the order lookup tool — likely the tool is returning an error or empty response that the agent interprets as "no order found," triggering re-prompting  
- C. Increase the agent's context window  
- D. The agent's IAM role lacks permission to Lambda  

**Answer: B**  
*Debug: Agent traces reveal the exact tool response. The Lambda function likely returns a status code 200 but with an empty body or error message the agent misinterprets. Fix the Lambda return format.*

---

**Q14.** A Bedrock Agent calling an external payment API is producing "Error: Tool execution failed" after exactly 29 seconds. This happens consistently.

What is the root cause?

- A. The Bedrock Agent has a 30-second hard timeout on tool calls  
- B. The Lambda function backing the action group has a 30-second timeout. The payment API call takes longer than 29 seconds, causing Lambda to time out  
- C. The payment API rejects calls over 29 seconds  
- D. Bedrock Guardrails is blocking the payment tool call  

**Answer: B**  
*Debug: Increase Lambda function timeout (max 15 minutes). For genuinely long-running operations, implement an async pattern: Lambda initiates the payment, stores job ID, Agent polls a status-check tool.*

---

**Q15.** A Step Functions workflow orchestrating multiple FM calls runs successfully in development but fails in production with "States.TaskFailed" on the Bedrock invocation step.

What is the FIRST thing to check?

- A. The Step Functions state machine definition has a bug  
- B. The IAM execution role for Step Functions lacks `bedrock:InvokeModel` permission on the specific model ARN being called in production  
- C. The Bedrock model is not available in the production region  
- D. The Step Functions timeout is too short  

**Answer: B**  
*Debug: IAM permissions in dev and prod environments may differ. Check the Step Functions execution role policies. Verify the model ARN in the `Resource` field of the IAM policy matches what's being called.*

---

**Q16.** A multi-agent system where a supervisor delegates to specialist agents works for simple tasks but "forgets" context from earlier in the conversation when tasks span more than 3-4 turns.

What is the architectural issue?

- A. The agents are hitting their context window limit  
- B. Each agent invocation starts without access to prior conversation context because shared state is not being persisted. Solution: Implement AgentCore Memory to maintain shared context across agent invocations  
- C. The supervisor agent's IAM role is missing  
- D. The specialist agents have conflicting instructions  

**Answer: B**  
*Debug: Without persistent memory, each agent turn is stateless. AgentCore Memory or external DynamoDB session storage must be explicitly implemented to share conversation history.*

---

**Q17.** An API Gateway → Lambda → Bedrock streaming implementation works correctly on localhost but produces "502 Bad Gateway" errors when deployed to API Gateway.

What is the cause?

- A. Bedrock streaming is not supported in the deployed region  
- B. API Gateway REST API does not support response streaming. The solution is to use API Gateway WebSocket API or HTTP API with Lambda streaming response  
- C. The Lambda function has incorrect IAM permissions  
- D. The CORS configuration is missing  

**Answer: B**  
*Debug: REST API requires complete response before returning to client. For streaming, use WebSocket API (bidirectional) or HTTP API with payload format 2.0 and Lambda streaming.*

---

**Q18.** A Bedrock Flows execution always takes the same path regardless of the Condition node's input, ignoring the branching logic.

What is wrong?

- A. Condition nodes in Bedrock Flows only support one output path  
- B. The condition expression references a variable name that doesn't match the actual output key from the preceding node — the condition evaluates on null/undefined and always takes the default path  
- C. The foundation model in the preceding node returns responses in wrong format  
- D. Bedrock Flows doesn't support conditional branching  

**Answer: B**  
*Debug: Check the exact output schema from the preceding node in the trace. Condition node variable references must exactly match the key names in the upstream node's output.*

---

**Q19.** A circuit breaker implemented in Step Functions is supposed to return a fallback response when Bedrock fails, but instead the entire workflow fails with an unhandled error.

What is the fix?

- A. Increase the Step Functions timeout  
- B. The Catch block in the Bedrock Task state is missing or has a typo in the ErrorEquals field. Add `"Catch": [{"ErrorEquals": ["States.ALL"], "Next": "FallbackState"}]`  
- C. Lambda needs more memory  
- D. Add a Retry block with exponential backoff  

**Answer: B**  
*Debug: Examine the state machine definition. The `Catch` configuration must correctly specify which errors to catch. `States.ALL` catches any error. A typo in the state name or missing Catch means errors propagate uncaught.*

---

**Q20.** An agent designed to answer questions about internal company policies routes ALL questions to the knowledge base, even simple math questions like "What is 15% of $200?"

What is wrong and how do you fix it?

- A. The knowledge base contains math documents  
- B. The agent's instruction prompt is too broad — it doesn't specify when to use the knowledge base vs. when to answer directly from model knowledge. Refine instructions to specify: "Only query the knowledge base for policy-related questions"  
- C. The math tool is not configured  
- D. Bedrock Agents cannot do math  

**Answer: B**  
*Debug: Review agent instructions for specificity. Add explicit guidance on when each tool should be used. This prevents unnecessary KB calls and reduces costs.*

---

**Q21.** A human-in-the-loop Step Functions workflow sends approval requests to SQS but the workflow never resumes after a human approves via a web UI that calls a Lambda function.

What is missing?

- A. The SQS queue needs a DLQ configured  
- B. The Lambda function called by the UI is not calling `step_functions.send_task_success(taskToken=token, output=json.dumps(result))` with the original task token from the SQS message  
- C. The IAM role needs SQS permissions  
- D. The Step Functions execution expired  

**Answer: B**  
*Debug: Task Token pattern requires explicitly calling `SendTaskSuccess` or `SendTaskFailure` with the original token. The human UI Lambda must extract the token from the SQS message and send it back.*

---

**Q22.** An agentic system using MCP shows tools working in development (deployed to Lambda) but missing in production (where they're deployed to ECS).

What should be checked?

- A. ECS doesn't support MCP  
- B. The MCP server endpoint URL registered in the AgentCore Gateway may point to the development Lambda ARN instead of the production ECS service endpoint  
- C. MCP tools are cached by Bedrock and not refreshed  
- D. ECS requires a different MCP client version  

**Answer: B**  
*Debug: AgentCore Gateway stores the MCP server endpoint. In production, verify the registered endpoint points to the correct ECS service URL/ALB, not the dev Lambda function.*

---

### SECTION C — Guardrails, Security & Access Failures (Q23–Q30)

**Q23.** Bedrock Guardrails is configured to block competitor mentions, but when a user asks "Compare AWS with Azure for machine learning", the response still mentions Azure.

What is the issue?

- A. Guardrails cannot block proper nouns  
- B. The word filter is configured with BLOCK action on input only, not on output. The model generates the comparison and Azure appears in the OUTPUT side which is not filtered  
- C. The guardrail was not attached to the invocation  
- D. Azure is not in the word filter list  

**Answer: B or C** *(most likely C in exam context)*  
*Debug: First verify the guardrail is correctly attached to the Bedrock invocation using `guardrailIdentifier` and `guardrailVersion` parameters. Then verify OUTPUT filtering is also enabled for the word.*

---

**Q24.** Lambda in a private subnet cannot reach Amazon Bedrock, returning `Unable to connect to endpoint` errors. The Lambda has the correct IAM permissions.

What is missing?

- A. Lambda needs a public IP address  
- B. An Interface VPC Endpoint for `com.amazonaws.REGION.bedrock-runtime` is missing in the VPC. Lambdas in private subnets have no internet access and need a VPC endpoint to reach Bedrock  
- C. Bedrock is not available in this region  
- D. Lambda needs a NAT Gateway  

**Answer: B**  
*Debug: Private subnets have no internet access. Two options: (1) NAT Gateway (but this routes through internet — fails the "private" requirement) or (2) Interface VPC Endpoint for Bedrock Runtime (fully private).*

---

**Q25.** CloudTrail logs show `bedrock:InvokeModel` calls from an application, but the logs show `userIdentity.type = "Root"` instead of the expected Lambda IAM role.

What is the security problem?

- A. CloudTrail is not configured correctly  
- B. The application is using root account credentials (hardcoded or in environment variables) instead of an IAM role. This is a critical security violation  
- C. The Lambda function needs more IAM permissions  
- D. Bedrock requires root credentials for some operations  

**Answer: B**  
*Debug: Root credentials should NEVER be used for application access. Assign an IAM execution role to the Lambda function and ensure it uses instance metadata (not hardcoded keys) for credential retrieval.*

---

**Q26.** A company's RAG system is supposed to enforce row-level security — Finance team only sees finance documents, HR only sees HR documents. But all users are seeing all documents.

What is wrong?

- A. OpenSearch doesn't support row-level security  
- B. The metadata filter is not being dynamically constructed from the user's session/JWT claims. The application is calling Retrieve with no filter, returning all documents to all users  
- C. The documents were not tagged with department metadata during ingestion  
- D. The knowledge base IAM policy is too permissive  

**Answer: B** *(C is also possible — debug both)*  
*Debug: Check two things: (1) Documents have correct department metadata during ingestion. (2) The Lambda function dynamically builds the metadata filter from the authenticated user's claims before calling Retrieve.*

---

**Q27.** PII (Social Security Numbers) is appearing in CloudWatch logs despite having Bedrock Guardrails with PII filtering enabled.

What is the configuration error?

- A. CloudWatch doesn't support PII filtering  
- B. Guardrails is enabled for model invocations but Model Invocation Logging is writing the ORIGINAL request before guardrails processes it. Guardrail PII masking must be applied to the logging path as well  
- C. SSN is not in the supported PII entity types  
- D. The guardrail version is outdated  

**Answer: B**  
*Debug: Ensure Guardrails is applied BEFORE the request reaches the logging pipeline, or verify that the logging configuration captures the POST-guardrail payload. The correct order: Request → Guardrails → [masked payload] → Model → [masked response] → Logs.*

---

**Q28.** A SageMaker model evaluation shows 40% demographic disparity in recommendation quality between age groups, but no CloudWatch alarm fired.

What is wrong?

- A. SageMaker Clarify doesn't support age-based fairness metrics  
- B. The CloudWatch alarm was configured to monitor the wrong metric namespace or dimension, OR the alarm threshold is set higher than 40%. Check that the alarm monitors the correct Clarify fairness metric and that the threshold is set to 15% as intended  
- C. The evaluation is using the wrong test dataset  
- D. Model Monitor is not enabled  

**Answer: B**  
*Debug: Verify (1) CloudWatch alarm metric source matches the Clarify output metric name exactly, (2) alarm threshold is set to the intended value (15%), (3) Model Monitor is in ENABLED state and Clarify processor is attached.*

---

**Q29.** A cross-account Lake Formation setup grants column-level access to an application Lambda, but Athena queries still return all columns including restricted ones.

What is the issue?

- A. Lake Formation doesn't support cross-account Athena queries  
- B. The Athena execution role is using its own IAM permissions (which may be over-permissive) rather than going through Lake Formation permission checks. Verify Lake Formation is registered as the authoritative access control for the Glue catalog and the Athena workgroup is configured to use Lake Formation  
- C. Athena has a known bug with column filtering  
- D. The column grant needs to be re-applied  

**Answer: B**  
*Debug: Lake Formation must be configured as the authorization model for the catalog. Check that "Use only IAM access control" is NOT checked in the Lake Formation settings — Lake Formation permissions must take precedence.*

---

**Q30.** A Lambda-based GenAI API is passing security scanning but S3 buckets containing vector embeddings are publicly readable by anyone with the S3 URL.

What went wrong?

- A. OpenSearch Serverless is exposing embeddings to S3  
- B. The S3 bucket Block Public Access setting was not enabled, and a bucket policy or ACL was misconfigured to allow public read — possibly from a developer testing phase that wasn't reverted  
- C. Bedrock automatically makes embedding vectors public  
- D. The VPC endpoint configuration exposes the bucket  

**Answer: B**  
*Debug: Enable S3 Block Public Access at the account level (catches all future misconfigurations). Audit existing bucket ACLs and policies. Use Amazon Macie to scan for sensitive data in public buckets.*

---

### SECTION D — Performance, Cost & Operational Failures (Q31–40)

**Q31.** A production Bedrock API shows 3-second cold starts affecting every first request in the morning when the service wakes up from overnight inactivity.

What is the fix?

- A. Increase Lambda memory allocation  
- B. Enable Lambda Provisioned Concurrency to keep execution environments warm, eliminating cold starts  
- C. Use a larger instance type for Lambda  
- D. Reduce the function package size  

**Answer: B**  
*Debug: Cold starts occur when no warm execution environment is available. Provisioned Concurrency pre-initializes N execution environments that are always ready.*

---

**Q32.** Bedrock Batch Inference job was submitted 6 hours ago but is still "In Progress." The input file has 10,000 records. What should be investigated?

- A. Batch inference has a maximum of 5,000 records  
- B. Check (1) IAM role permissions for the batch job (missing S3 read/write), (2) S3 input file format (must be valid JSONL with correct schema), (3) CloudWatch logs for the job. Most commonly a permissions or malformed input issue  
- C. The model doesn't support batch inference  
- D. Increase the job timeout parameter  

**Answer: B**  
*Debug: In order: Check IAM role → Check JSONL format validity → Check CloudWatch logs for error details. Batch jobs fail silently if input format is wrong or permissions are insufficient.*

---

**Q33.** Embedding costs are $5,000/month. The team tries to reduce this by increasing chunk sizes from 300 to 900 tokens. After the change, the monthly cost drops to $3,000 but support tickets about wrong answers double.

What happened?

- A. The embedding model has a known bug with large chunks  
- B. Larger chunks = fewer total embeddings = lower cost, BUT semantic dilution degrades retrieval quality, causing more wrong answers. This is a cost-quality trade-off made in the wrong direction  
- C. The vector store can't index large chunks correctly  
- D. The foundation model is hallucinating more  

**Answer: B**  
*Fix: Revert chunk size. Reduce embedding dimensions instead (e.g., from 1024 to 512 for Titan) — this reduces cost without harming retrieval precision.*

---

**Q34.** An application switches model IDs using Lambda environment variables. During a model switch, half the users get responses from the old model and half from the new one for several minutes.

What architectural problem exists?

- A. Lambda environment variables cannot store model IDs  
- B. Lambda environment variable updates propagate gradually across running instances — during the propagation window, different instances use different configs. AWS AppConfig with instant propagation solves this  
- C. The Lambda has too many concurrent instances  
- D. Bedrock caches requests by model ID  

**Answer: B**  
*Fix: Use AWS AppConfig. Lambda instances fetch config from AppConfig on each request (cached by the AppConfig Agent extension). Updates propagate within seconds to all instances simultaneously.*

---

**Q35.** A developer uses `response_hash = hashlib.sha256(llm_response.encode()).hexdigest()` to detect when Bedrock's response changes. They find the hash changes on every API call, even for the same question.

What is the explanation?

- A. SHA-256 is non-deterministic  
- B. The LLM's temperature is > 0, enabling probabilistic sampling. Each call produces slightly different wording even for the same correct answer, resulting in different hashes  
- C. Bedrock adds timestamps to responses  
- D. The model was updated between calls  

**Answer: B**  
*Debug: Set temperature=0 for deterministic outputs IF appropriate for the use case. For evaluation, use semantic evaluation metrics (Bedrock Model Evaluations) rather than hash comparison.*

---

**Q36.** An OpenSearch Serverless vector search that was fast (50ms) at 100K vectors is now slow (8,000ms) at 10M vectors. Nothing else changed.

What is the likely cause?

- A. OpenSearch Serverless doesn't scale beyond 1M vectors  
- B. The index grew into millions of small shards. As the index scaled, the shard count exploded with default auto-sharding, and each query now coordinates across thousands of tiny shards — massive coordination overhead  
- C. The embedding model changed  
- D. Network latency increased at scale  

**Answer: B**  
*Fix: Force merge shards and reconfigure the index with explicit shard count (targeting 30-50 GB per shard). For 10M vectors at 4KB each, that's ~40 GB → 1-2 primary shards is optimal.*

---

**Q37.** A company's Bedrock costs are $80,000/month. Cost Allocation Tags show the breakdown is impossible — all costs appear under a single "untagged" bucket.

What went wrong?

- A. Bedrock doesn't support cost allocation tags  
- B. Cost allocation tags were not applied to Bedrock Knowledge Bases, Agents, and invocation logging configurations during resource creation. Tags must be applied at resource creation time and activated in the AWS Billing console  
- C. Tags require AWS Enterprise Support to activate  
- D. Cost Explorer doesn't support Bedrock  

**Answer: B**  
*Fix: (1) Tag all Bedrock resources (team, app, environment), (2) Activate user-defined cost allocation tags in AWS Billing console, (3) Use Model Invocation Logs with Athena for per-request attribution.*

---

**Q38.** A streaming chat application works correctly for responses under 10 seconds but times out for complex questions that take 35–45 seconds to generate.

The architecture is: API Gateway REST API → Lambda → Bedrock InvokeModelWithResponseStream.

What is the architectural flaw?

- A. Lambda can't handle streaming responses  
- B. API Gateway REST API has a 29-second integration timeout that cannot be increased. For long streaming responses, use API Gateway WebSocket API or Lambda Function URLs with streaming  
- C. Bedrock streaming has a 30-second maximum  
- D. Lambda has a 30-second max execution time  

**Answer: B**  
*Fix: Replace REST API with WebSocket API (supports persistent connections for streaming) or use Lambda Function URL with response streaming enabled (bypasses API Gateway timeout).*

---

**Q39.** A CI/CD pipeline with Bedrock model evaluations consistently passes in staging but quality regressions go undetected in production. The evaluation job uses the same test dataset.

What is the gap?

- A. The evaluation job is not running in production  
- B. The test dataset does not reflect production query distribution. The staging dataset contains simple/expected queries while production gets edge cases, slang, multi-language queries, and complex scenarios not covered in the test set  
- C. The evaluation threshold is too strict  
- D. CloudWatch alarms are not configured for production  

**Answer: B**  
*Fix: Augment the regression test dataset with real production queries (anonymized). Include edge cases, failure modes discovered in production, and diverse query types. Update the dataset periodically.*

---

**Q40.** A developer deployed a new Bedrock Agent version. User complaints spike immediately. The CodeDeploy canary deployment shifted 100% of traffic before the CloudWatch alarm triggered.

What went wrong?

- A. The alarm metric was monitoring the wrong resource  
- B. The CodeDeploy deployment configuration shifted traffic too quickly — canary interval was set to 1 minute, which was not long enough for errors to accumulate and breach the alarm threshold before 100% traffic shift  
- C. CloudWatch alarms don't integrate with CodeDeploy  
- D. The Lambda function needs more memory for the new agent version  

**Answer: B**  
*Fix: Increase canary interval (e.g., 10-15 minutes). Lower the alarm threshold to detect errors earlier. Use Linear deployment instead of Canary for even more gradual rollout (e.g., 10% per 5 minutes).*

---

## 🔧 Debugging Decision Tree

```
Symptom → First Check → Root Cause Category

Wrong answers in RAG → Faithfulness low? → Model hallucinating (enable grounding check)
Wrong answers in RAG → Context Recall low? → Retrieval missing docs (check chunking/metadata)
Wrong answers in RAG → Correctness low, Faithfulness high → KB has stale data (re-ingest)

Agent stuck in loop → Enable traces → Check tool observation format
Agent wrong behavior → Check IAM permissions → Verify tool schema/OpenAPI spec

High costs → Check token counts → Model too large? Enable intelligent routing
High costs → Check batch eligibility → Use batch inference for time-tolerant tasks
High latency → X-Ray trace → Identify bottleneck (cold start/model/KB retrieval)

Security event → CloudTrail → Who/when/what
PII in logs → Verify guardrail attachment → Check apply order (before logging)
Data exposed → S3 Block Public Access → Check bucket policy + ACLs
```

---

## 📊 Debugging Coverage Map

| Area | Questions |
|------|-----------|
| RAG System Failures | Q1–Q12 |
| Agent & Orchestration Failures | Q13–Q22 |
| Security & Access Failures | Q23–Q30 |
| Performance & Cost Failures | Q31–Q40 |
