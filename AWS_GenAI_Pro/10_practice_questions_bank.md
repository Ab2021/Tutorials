# Practice Questions Bank – 60+ Scenario Questions
## AIP-C01 – All Domains Covered

---

## 🎯 How to Use This Bank

- Read every **explanation** — the reasoning matters more than the answer
- Focus on **why wrong answers are wrong** (elimination is key)
- Map each question to its **exam domain**

---

## Domain 1 – Foundation Model Integration & Data Management

**Q1.** A retail company must embed 10 million product descriptions. Storage costs are high and queries are slow. Without changing the embedding model, what should they do?

- A. Remove older product descriptions to reduce index size  
- B. Switch to a third-party embedding model with better compression  
- C. Reconfigure Amazon Titan Embeddings V2 to use lower dimensionality (e.g., 384 or 512)  
- D. Increase chunk size to reduce total embeddings  

**Answer: C**  
*Reasoning: Titan Embeddings V2 supports configurable output dimensions. Reducing from 1024→512 halves storage and speeds up distance calculations. Removing data reduces value; switching models adds migration cost; larger chunks hurt retrieval precision.*

---

**Q2.** A medical company needs FMs to understand proprietary drug interaction terminology not in general training data. Which customization is MOST appropriate?

- A. Prompt engineering with terminology glossary in system prompt  
- B. RAG with a terminology knowledge base  
- C. Continued pre-training on clinical literature  
- D. Supervised fine-tuning on Q&A pairs  

**Answer: C**  
*Reasoning: Continued pre-training injects domain vocabulary into model weights. RAG is for dynamic lookup not vocabulary internalization. Prompting has token limits. Fine-tuning teaches style/format not new vocabulary.*

---

**Q3.** A company's RAG system retrieves correct documents but the model lacks enough surrounding context to generate complete answers. What chunking strategy fixes this?

- A. Fixed-size chunking with overlap  
- B. Semantic chunking  
- C. Hierarchical chunking (small child for retrieval, large parent for context)  
- D. No chunking (full document)  

**Answer: C**  
*Reasoning: Hierarchical chunking uses small children for precise retrieval and returns the larger parent chunk to the model, providing both accuracy and context.*

---

**Q4.** A company wants to automatically track which S3 documents contributed to specific AI responses for compliance audits. Which combination achieves TWO-LAYER traceability?

- A. CloudTrail (who called) + Glue Data Catalog metadata in vector store (which document)  
- B. CloudWatch metrics + S3 access logs  
- C. X-Ray traces + DynamoDB session logs  
- D. Model Invocation Logs + S3 versioning  

**Answer: A**  
*Reasoning: CloudTrail = WHO (IAM principal, API call). Glue Data Catalog metadata propagated to vector store = WHICH document was retrieved.*

---

**Q5.** A company's Bedrock Knowledge Base keeps returning outdated policy documents even though they've been updated in S3. What is the root cause?

- A. The embedding model needs retraining  
- B. Metadata filters are missing, misconfigured, or not applied at query time  
- C. The foundation model is using cached responses  
- D. The chunk size is too large  

**Answer: B**  
*Reasoning: Outdated documents appear when metadata filtering logic is absent or broken. Documents must be tagged (e.g., effective_date) and filters applied at retrieval time.*

---

**Q6.** Which document pre-processing service converts complex PDFs with tables and images into structured JSON suitable for Bedrock Knowledge Base ingestion?

- A. Amazon Textract  
- B. Amazon Rekognition  
- C. Amazon Bedrock Data Automation (BDA)  
- D. AWS Glue DataBrew  

**Answer: C**  
*Reasoning: BDA provides semantic document understanding — it understands layout, preserves structure, and outputs JSON. Textract is for forms/tables OCR. Rekognition is for image object detection.*

---

**Q7.** What training data format does Amazon Bedrock require for supervised fine-tuning?

- A. CSV with headers  
- B. JSONL with prompt/completion pairs  
- C. Parquet files  
- D. XML with structured schema  

**Answer: B**  
*Reasoning: Bedrock fine-tuning uses JSONL format: `{"prompt": "...", "completion": "..."}` per line.*

---

## Domain 2 – Implementation and Integration

**Q8.** An application displays AI responses character-by-character in a React frontend. The app uses API Gateway. Which combination is correct?

- A. REST API + InvokeModel + client polls every 100ms  
- B. WebSocket API + Lambda + InvokeModelWithResponseStream  
- C. HTTP API + DynamoDB cache + paginated GET requests  
- D. Direct Bedrock calls from React using IAM user credentials  

**Answer: B**  
*Reasoning: WebSocket enables bidirectional streaming. InvokeModelWithResponseStream delivers tokens as generated. REST API waits for full response. IAM user credentials should never be exposed in browsers.*

---

**Q9.** A team has 20 existing REST APIs they want to expose as Bedrock Agent tools without writing custom Lambda wrappers. What is the best approach?

- A. Create 20 Lambda functions, one per API  
- B. Amazon Bedrock AgentCore Gateway with OpenAPI specifications  
- C. API Gateway as proxy with URLs referenced in agent instructions  
- D. Single Lambda with routing logic for all 20 APIs  

**Answer: B**  
*Reasoning: AgentCore Gateway auto-generates tool definitions from OpenAPI specs, eliminating manual code.*

---

**Q10.** Multiple specialized Bedrock Agents (research, finance, legal) all need access to the same conversation history across sessions. What manages this shared state?

- A. DynamoDB with shared session table  
- B. Amazon Bedrock AgentCore Memory  
- C. ElastiCache Redis  
- D. Step Functions with shared context  

**Answer: B**  
*Reasoning: AgentCore Memory is specifically designed for shared state across specialized agents in multi-agent architectures.*

---

**Q11.** A Bedrock Agent is stuck in an infinite loop calling the same tool repeatedly. What is the BEST first debugging step?

- A. Increase Lambda function timeout  
- B. Reduce the agent's context window  
- C. Enable Agent Traces to see the reasoning chain  
- D. Add CloudWatch alarms for high tool invocation counts  

**Answer: C**  
*Reasoning: Traces reveal EXACTLY which step causes the loop and why. Timeout/context changes mask symptoms. CloudWatch only shows THAT looping occurs, not WHY.*

---

**Q12.** A company needs to pause a workflow for human approval of high-risk AI decisions, then resume automatically after approval. What implements this?

- A. SQS with polling Lambda  
- B. Step Functions with Task Tokens  
- C. EventBridge with scheduled rules  
- D. SNS with email notifications + manual restart  

**Answer: B**  
*Reasoning: Task Tokens pause Step Functions execution (zero resource consumption) until the token is returned by the human approver, then resume with full state intact.*

---

**Q13.** The Converse API is returning errors. The developer is passing `messages` as a plain text string. What is the fix?

- A. Switch to InvokeModel which supports plain text  
- B. Structure messages as an array: `[{"role": "user", "content": [{"text": "..."}]}]`  
- C. Add Content-Type: text/plain header  
- D. URL-encode the message  

**Answer: B**  
*Reasoning: Converse API requires structured messages array with role and content objects. Plain text is not supported.*

---

**Q14.** An agent needs lightweight tools (Lambda-suitable) and CPU-intensive tools (containerized). What architecture handles both optimally?

- A. Deploy everything to ECS  
- B. Deploy everything to Lambda with increased memory  
- C. Hybrid: MCP with lightweight tools in Lambda, heavy tools in ECS  
- D. Use Bedrock Flows to separate the tool types  

**Answer: C**  
*Reasoning: MCP abstracts compute layer. Lambda handles lightweight tools cost-effectively; ECS handles CPU-intensive workloads without Lambda's limitations.*

---

**Q15.** An e-commerce company needs to route requests to different FMs based on user tier, cost thresholds (changing hourly), and regulatory zone — without deploying new code. Which solution is correct?

- A. Lambda with environment variables for routing rules  
- B. API Gateway stage variables for model endpoints  
- C. Lambda fetching routing config from AWS AppConfig Agent per request  
- D. Lambda authorizers evaluating routing rules stored in AppConfig  

**Answer: C**  
*Reasoning: AppConfig provides real-time configuration updates without code deployment. Lambda env vars require deployment to change. AppConfig Agent caches config locally in Lambda for performance.*

---

## Domain 3 – AI Safety, Security & Governance

**Q16.** A finance AI assistant must: (1) never discuss competitor products, (2) block stock recommendations, (3) only cite approved financial guidance. Choose THREE guardrail configurations.

- A. Add stock recommendations and guaranteed returns to denied topics  
- B. Configure content filter for high-risk patterns  
- C. Configure content filter for competitor names  
- D. Add competitor names as word filters with BLOCK action  
- E. Set low grounding score threshold  
- F. Set high grounding score threshold  

**Answer: A, D, F**  
*Reasoning: A=Denied topics blocks business-specific topics. D=Word filters specifically blocks brand names. F=High grounding ensures only document-backed claims. Content filters (B,C) are for harmful content categories not business policies. Low threshold (E) would allow ungrounded responses.*

---

**Q17.** Bedrock traffic from Lambda must never traverse the public internet. What is required?

- A. NAT Gateway for outbound traffic  
- B. Interface VPC Endpoints for Bedrock Runtime in the application VPC  
- C. CloudFront distribution in front of Bedrock  
- D. VPN connection to AWS backbone  

**Answer: B**  
*Reasoning: Interface VPC Endpoints (PrivateLink) route traffic entirely within AWS private network. NAT Gateway still routes to public Bedrock endpoints.*

---

**Q18.** A company must log all Bedrock prompts and responses for compliance but cannot store raw PII. What combination achieves this?

- A. Model Invocation Logging to S3 + Amazon Macie for PII deletion  
- B. Bedrock Guardrails with PII masking + Model Invocation Logging  
- C. CloudTrail + Lambda post-processing to redact PII  
- D. Kinesis Firehose with Lambda transformation  

**Answer: B**  
*Reasoning: Guardrails automatically masks/anonymizes PII BEFORE the log is written, ensuring PII never reaches storage. This is the correct order of operations.*

---

**Q19.** A company needs department-level model access control across 10,000 employees without managing individual IAM users. What is correct?

- A. Create IAM groups per department with inline policies  
- B. Use IAM Identity Center with department group to permission set mapping  
- C. Use S3 bucket policies to control which departments access training data  
- D. Configure Bedrock Guardrails per department  

**Answer: B**  
*Reasoning: IAM Identity Center maps SSO groups to permission sets with least-privilege IAM policies at the AWS resource layer — not application layer.*

---

**Q20.** Sensitive data must never physically leave the company's on-premises data center, even for AI processing. What architecture addresses this?

- A. VPC Endpoints for Bedrock + private subnets  
- B. AWS PrivateLink with transit gateway  
- C. AWS Outposts for local processing + de-identify before sending to Bedrock  
- D. Cross-region inference with EU-scoped profile  

**Answer: C**  
*Reasoning: VPC endpoints secure transit but compute runs in AWS DCs. Outposts places AWS infrastructure on-premises — data never leaves the facility. De-identification removes sensitive data before regional Bedrock call.*

---

**Q21.** A company must demonstrate immutability of training data in S3 for regulatory audits. What combination is correct?

- A. S3 versioning + MFA Delete  
- B. S3 Object Lock (COMPLIANCE mode) + CloudTrail S3 data events  
- C. AWS Backup + S3 replication  
- D. S3 Glacier + Vault Lock  

**Answer: B**  
*Reasoning: Object Lock COMPLIANCE mode prevents deletion/modification even by root. CloudTrail data events record every access and modification attempt for audit trail.*

---

**Q22.** A team wants to systematically validate that prompt injection attacks are blocked before each production deployment. What implements this?

- A. Manual review of 10 adversarial prompts per release  
- B. Automated adversarial prompt test suite executed via Step Functions in CI/CD  
- C. Configure Bedrock Guardrails and rely on them without testing  
- D. Penetration test quarterly by a security team  

**Answer: B**  
*Reasoning: Automated adversarial testing in CI/CD ensures defenses are verified on every deployment, not just periodically.*

---

**Q23.** A company runs continuous bias detection on a production classification model. Which service combination is correct?

- A. CloudWatch + CloudTrail  
- B. SageMaker Model Monitor + SageMaker Clarify + CloudWatch Alarms  
- C. Amazon A2I + SNS notifications  
- D. Bedrock Model Evaluations + EventBridge  

**Answer: B**  
*Reasoning: Model Monitor continuously captures predictions. Clarify computes fairness metrics. CloudWatch Alarms alert when bias exceeds thresholds.*

---

## Domain 4 – Operational Efficiency & Cost Optimization

**Q24.** A company processes 50 million product embeddings nightly and needs to minimize costs. 24-hour latency is acceptable. What is correct?

- A. Lambda with 100 concurrent executions  
- B. ECS with auto-scaling  
- C. Amazon Bedrock batch inference  
- D. Provisioned Throughput with maximum model units  

**Answer: C**  
*Reasoning: Batch inference is purpose-built for high-volume, time-tolerant workloads at ~50% cost reduction versus on-demand.*

---

**Q25.** A chatbot has a 2,000-token static system prompt and 1,000-token few-shot examples reused across all requests. How can token costs be reduced by up to 90% for this portion?

- A. Reduce few-shot examples to 1  
- B. Enable prompt caching with cache checkpoint after the few-shot examples  
- C. Switch to a smaller model  
- D. Implement conversation summarization  

**Answer: B**  
*Reasoning: 3,000 static tokens cached → subsequent requests pay ~10% for that portion. Cache checkpoint must be placed after the static content.*

---

**Q26.** A company serves 70% simple FAQ queries and 30% complex analytical tasks. They want to minimize cost without quality degradation for complex tasks. What is correct?

- A. Serve all traffic with Claude Haiku  
- B. Implement Bedrock Intelligent Prompt Routing  
- C. Use API Gateway stage variables to route by URL path  
- D. Add complexity classifier Lambda before all Bedrock calls  

**Answer: B**  
*Reasoning: Intelligent Prompt Routing automatically routes based on complexity — no client-side code changes required.*

---

**Q27.** Top 5% of users drive 60% of token costs by sending long, unique prompts. What controls cost for these users specifically without penalizing others?

- A. Global rate limits on the API  
- B. Switch to a smaller model for all users  
- C. Per-user token budgets using API Gateway usage plans + CountTokens API  
- D. Enable prompt caching  

**Answer: C**  
*Reasoning: Caching doesn't help for unique prompts. Global limits penalize all users. CountTokens measures consumption; usage plans enforce per-user limits.*

---

**Q28.** A company's OpenSearch vector index has millions of small shards, causing high query latency. What is the recommended fix?

- A. Add more OpenSearch nodes  
- B. Increase embedding dimensions to 1536  
- C. Consolidate to fewer, larger shards (30-50 GB each)  
- D. Switch to Aurora pgvector  

**Answer: C**  
*Reasoning: Small shards → each runs its own ANN search → massive coordination overhead. Consolidating reduces coordination and improves HNSW cache locality.*

---

**Q29.** A GenAI API needs sub-100ms TTFT for interactive chat users. Lambda cold starts add 500ms. What eliminates cold starts?

- A. Increase Lambda memory to 3GB  
- B. Use Lambda Provisioned Concurrency  
- C. Deploy Lambda to multiple regions  
- D. Use Container Images instead of zip deployment  

**Answer: B**  
*Reasoning: Provisioned Concurrency keeps Lambda execution environments initialized and warm, eliminating cold start latency.*

---

**Q30.** During a Bedrock service degradation, the application keeps retrying and worsening the outage. What pattern prevents this?

- A. Increase retry attempts with shorter timeouts  
- B. Implement Circuit Breaker pattern via Step Functions  
- C. Add CloudWatch alarms to detect failures  
- D. Route all traffic to a backup region  

**Answer: B**  
*Reasoning: Circuit Breaker detects failure threshold, opens circuit, returns fallback immediately (fail fast), and prevents retry storms that worsen outages.*

---

## Domain 5 – Testing, Validation & Troubleshooting

**Q31.** A RAG app returns correct answers but cites irrelevant documents. Which metrics diagnose this?

- A. Faithfulness and Coherence  
- B. Citation Precision and Context Relevance  
- C. ROUGE and BLEU scores  
- D. Embedding cosine similarity  

**Answer: B**  
*Reasoning: Citation precision measures whether citations support claims. Context relevance measures whether retrieved passages are relevant to the query.*

---

**Q32.** Why does response hashing fail as a RAG regression testing approach?

- A. Hash collisions are too common  
- B. LLMs are non-deterministic; same correct answer produces different hashes each run  
- C. Hashing is too computationally expensive  
- D. Response hashes change when the knowledge base updates  

**Answer: B**  
*Reasoning: Temperature > 0 means LLMs sample probabilistically. Different words, same correct answer = different hash = 100% false positive regression detection rate.*

---

**Q33.** A RAG system starts producing wrong answers after a knowledge base update. What is the MOST reliable detection mechanism?

- A. Monitor user satisfaction scores  
- B. Compare embedding similarity of old vs. new responses  
- C. Maintain a fixed regression test dataset and run Bedrock model evaluations after every ingestion  
- D. Hash responses before and after update  

**Answer: C**  
*Reasoning: Fixed test dataset + correctness/completeness metrics is the only reliable approach. User satisfaction is a lagging indicator. Embedding similarity and hashing both fail for non-deterministic outputs.*

---

**Q34.** Two prompt variants need to be compared for accuracy before production. What is the SAFEST approach?

- A. A/B test on 10% of live production traffic  
- B. Bedrock Model Evaluations with Compare feature on a fixed test dataset  
- C. Manual review of 20 responses per variant  
- D. Deploy both and compare CloudWatch error rates  

**Answer: B**  
*Reasoning: Pre-production offline evaluation is safer than A/B testing live users. More scalable and consistent than manual review.*

---

**Q35.** A developer needs to identify whether high latency in a Bedrock agent comes from the model, the Lambda tool execution, or cold starts. Which service provides this granularity?

- A. Amazon CloudWatch metrics  
- B. AWS X-Ray with subsegments for each pipeline stage  
- C. Bedrock Model Invocation Logs  
- D. AWS CloudTrail  

**Answer: B**  
*Reasoning: X-Ray provides per-subsegment timing across services in a single trace. CloudWatch only shows aggregate metrics. CloudTrail is for audit, not performance.*

---

**Q36.** Noisy transcripts from Amazon Transcribe (with fillers like "um", "uh") are being sent directly to Bedrock, causing poor responses. What is the recommended fix?

- A. Fine-tune the foundation model on noisy transcripts  
- B. Increase the model temperature to handle noise  
- C. Use Lambda + Amazon Comprehend to normalize text at the source before inference  
- D. Add a Guardrail to filter filler words  

**Answer: C**  
*Reasoning: Clean at source is the architectural principle. Fine-tuning on noisy data teaches the model to accept noise. Comprehend detects entities for normalization context; custom Lambda removes fillers.*

---

**Q37.** A company needs automatic rollback if a new Lambda deployment causes elevated error rates. What implements this?

- A. CloudWatch alarms + manual rollback procedure  
- B. AWS CodeDeploy with Canary or Linear traffic shifting + CloudWatch alarm-triggered auto-rollback  
- C. AWS Config rules for deployment compliance  
- D. Step Functions with retry on failure  

**Answer: B**  
*Reasoning: CodeDeploy canary routing sends partial traffic to new version; if error rate exceeds alarm threshold, CodeDeploy automatically rolls back.*

---

**Q38.** A company's AI assistant for medical queries is both blocking legitimate medical education questions AND failing to block harmful advice. What is the BEST approach?

- A. Disable all guardrails and rely on the model's built-in safety  
- B. Configure Bedrock Guardrails to filter harmful content while ALLOWING educational health topics; enable PII masking  
- C. Route all medical queries to a human agent  
- D. Use denied topics to block all health-related content  

**Answer: B**  
*Reasoning: Balance safety and utility. Guardrails can allow legitimate educational health content while blocking harmful advice. Blanket blocking all health content is non-scalable and unhelpful.*

---

## Advanced Scenario Questions

**Q39.** A company builds a customer support app with: React frontend, AWS Amplify, AppSync GraphQL, Bedrock Knowledge Bases. Users report frequent timeouts on complex questions. The Lambda resolver uses RequestResponse invocation. What fixes this?

- A. Use AWS Amplify AI Kit for streaming GraphQL responses  
- B. Increase Lambda timeout to 15 minutes  
- C. Switch to SQS between AppSync and Lambda  
- D. Use API Gateway REST API instead of AppSync  

**Answer: A**  
*Reasoning: Amplify AI Kit implements streaming responses through GraphQL subscriptions, eliminating timeouts without increasing limits or changing architecture.*

---

**Q40.** A financial company's AI assistant must: generate responses in consistent format, adapt tone per business unit (legal/HR/finance), block hate speech and PHI, and allow adjustable content moderation over time — with LEAST maintenance overhead.

- A. Bedrock Prompt Management with variants per BU + Bedrock Guardrails with category filters and adjustable sensitive term lists  
- B. DynamoDB for prompts + Step Functions for validation + Comprehend for content filtering  
- C. Bedrock with system prompt injection per BU + Lambda for post-processing  
- D. SageMaker Canvas with S3-stored prompt templates  

**Answer: A**  
*Reasoning: Prompt Management handles centralized templates + variants per BU. Guardrails handle content moderation with adjustable thresholds. No custom orchestration code needed.*

---

**Q41.** A global e-commerce company needs to route Bedrock model calls based on: user tier, transaction value, regulatory zone, and hourly cost metrics — with immediate propagation to thousands of concurrent Lambdas and no code deployment.

- A. Lambda environment variables updated via console  
- B. API Gateway stage variables  
- C. Lambda fetching routing config from AWS AppConfig Agent each request  
- D. Lambda authorizers with AppConfig routing returning auth contexts  

**Answer: C**  
*Reasoning: AppConfig Agent runs as a Lambda extension, caching config locally. Config updates propagate immediately (within seconds) without Lambda redeployment. Far superior to env vars which require deployment.*

---

**Q42.** A company needs a financial services vector search solution supporting: 10M multilingual embeddings (English, Spanish, Portuguese), metadata filtering by date/agency/type, low latency, and minimal operational overhead.

- A. Amazon OpenSearch Serverless with Bedrock Knowledge Bases  
- B. Aurora PostgreSQL + pgvector with SQL similarity search  
- C. Amazon S3 Vectors with non-filterable metadata  
- D. Amazon Neptune Analytics with vector index  

**Answer: A**  
*Reasoning: OpenSearch Serverless: handles 10M+ embeddings, multilingual (embedding model handles multilingual), rich metadata filtering, low latency, fully managed (minimal ops overhead). S3 Vectors has limited filtering. Aurora requires more management.*

---

**Q43.** A multi-account AWS setup: application account has Lambda in private VPC calling Bedrock. Data lake is in a centralized storage account with sensitive columns. Requirements: private Bedrock connectivity + column-level cross-account access.

- A. Interface VPC Endpoints for Bedrock + Lake Formation LF-tag column grants  
- B. NAT Gateway + S3 bucket policies  
- C. Gateway endpoint for S3 + IAM path-based policies  
- D. VPC Endpoints + IAM path-based policies only  

**Answer: A**  
*Reasoning: Interface VPC Endpoints = private Bedrock connectivity. Lake Formation LF-tag-based column grants = fine-grained cross-account column-level access. This is the only option satisfying BOTH requirements.*

---

**Q44.** A media company managing hundreds of prompt templates across multiple Regions needs: version control, approval workflows with notifications, audit trails, and consistent parameterization. Which solution meets ALL requirements?

- A. Bedrock Studio prompt templates + CloudWatch dashboards + DynamoDB approval status  
- B. Bedrock Prompt Management (version control) + CloudTrail (audit) + IAM (approval control) + parameterized variables  
- C. Step Functions approval workflow + S3 prompt storage + EventBridge notifications  
- D. SageMaker Canvas + CloudFormation + AWS Config  

**Answer: B**  
*Reasoning: Prompt Management has built-in versioning, parameterization, and IAM-controlled approvals. CloudTrail provides automatic audit logging. This is the purpose-built solution.*

---

**Q45.** A customer support app needs to display AI-generated responses character by character, support thousands of concurrent users, with 15-45 second typical response times.

- A. API Gateway WebSocket API + Lambda + InvokeModelWithResponseStream  
- B. REST API + Lambda + standard InvokeModel + client polling every 100ms  
- C. Direct frontend connections to Bedrock with IAM user credentials  
- D. HTTP API + DynamoDB response cache + paginated GET requests  

**Answer: A**  
*Reasoning: WebSocket enables bidirectional persistent connections. InvokeModelWithResponseStream delivers tokens as generated. Supports thousands of concurrent connections. REST API doesn't support streaming. Direct IAM user credentials in frontend = security risk.*

---

## 🎯 Quick Reference: Common Exam Traps

| Trap | Correct Answer |
|------|---------------|
| "Fine-tune on noisy data" | Clean at source with Comprehend |
| "Response hashing for RAG testing" | Fixed test dataset + model evaluations |
| "Increase chunk size to reduce costs" | Reduce embedding dimensionality |
| "CloudWatch shows loop occurring" | Enable Agent Traces to see WHY |
| "Lambda env vars for model switching" | AWS AppConfig |
| "S3 bucket policies for RAG security" | Row-level metadata filters at query time |
| "High grounding threshold" | STRICTEST grounding (blocks ungrounded responses) |
| "Low grounding threshold" | Most permissive (allows ungrounded responses) |
| "Provisioned throughput for spiky traffic" | On-Demand for variable; Provisioned for sustained |
| "Embedding similarity for accuracy eval" | Correctness/Completeness metrics (ground truth) |
| "Increase context window for long chats" | Conversation summarization |
| "Add more OpenSearch nodes for shard latency" | Consolidate shards to 30-50 GB each |
| "Caching for power user cost control" | Per-user token budgets (API GW + CountTokens) |
| "Content filter for competitor names" | Word Filters with BLOCK action |
| "Denied topics for harmful content" | Content Filters for harmful; Denied Topics for business policy |

---

*End of Practice Questions Bank — Total: 45 scenario questions + 15 trap patterns*
