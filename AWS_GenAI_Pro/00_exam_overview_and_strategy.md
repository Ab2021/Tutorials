# AWS Certified Generative AI Developer – Professional (AIP-C01)
## Complete Exam Overview & Strategy Guide

---

## 📋 Exam At-a-Glance

| Attribute | Detail |
|-----------|--------|
| **Exam Code** | AIP-C01 |
| **Level** | Professional |
| **Scored Questions** | 65 |
| **Unscored (Experimental)** | 10 (not identified during exam) |
| **Total Questions** | 75 |
| **Time Limit** | 130 minutes |
| **Passing Score** | 750 / 1000 |
| **Question Format** | Single-answer MCQ + Multi-answer MCQ |
| **Price** | $300 USD |
| **Prerequisites** | None formal; 2+ years AWS + 1+ year GenAI recommended |

---

## 🎯 Five Exam Domains & Weightings

```
Domain 1: Foundation Model Integration, Data Management & Compliance  ████████████ 31%
Domain 2: Implementation and Integration                               ██████████  26%
Domain 3: AI Safety, Security, and Governance                         ████████    20%
Domain 4: Operational Efficiency and Optimization                     █████       12%
Domain 5: Testing, Validation, and Troubleshooting                    ████        11%
```

---

## 📚 Domain Breakdown

### Domain 1 – Foundation Model Integration, Data Management & Compliance (31%)

**Task Statements:**
- Select appropriate foundation models based on use-case requirements (latency, cost, modality)
- Design RAG pipelines with appropriate chunking, embedding, and retrieval strategies
- Prepare training/fine-tuning datasets; choose correct customization method
- Implement data governance, lineage, and compliance for AI workloads
- Manage vector stores and embedding lifecycle

**Key Services:**
- Amazon Bedrock (Knowledge Bases, Model Catalog)
- Amazon S3, AWS Glue, AWS Glue Data Catalog
- Amazon OpenSearch Serverless, Aurora PostgreSQL with pgvector
- Amazon Titan Embeddings, Cohere Embed, Nova Multimodal Embeddings
- Amazon Bedrock Data Automation (BDA)
- SageMaker Model Registry + Model Cards

---

### Domain 2 – Implementation and Integration (26%)

**Task Statements:**
- Build and orchestrate Bedrock Agents with action groups and tool use
- Integrate Knowledge Bases with RAG workflows
- Implement streaming, batch, and real-time inference patterns
- Configure Model Context Protocol (MCP) for tool portability
- Build multi-agent architectures with shared state management

**Key Services:**
- Amazon Bedrock Agents, AgentCore, AgentCore Gateway
- Amazon Bedrock Flows
- AWS Step Functions, AWS Lambda
- Amazon API Gateway (REST, HTTP, WebSocket)
- AWS AppSync, AWS Amplify
- Amazon Bedrock Prompt Management

---

### Domain 3 – AI Safety, Security, and Governance (20%)

**Task Statements:**
- Implement Bedrock Guardrails (content filters, PII masking, grounding, denied topics)
- Design defense-in-depth security architectures for GenAI workloads
- Enforce IAM-based access control for foundation models
- Implement private networking with VPC Endpoints and PrivateLink
- Address data residency requirements with Outposts
- Apply responsible AI principles; detect and mitigate bias

**Key Services:**
- Amazon Bedrock Guardrails
- AWS WAF, Amazon Comprehend
- AWS IAM, IAM Identity Center
- VPC Interface Endpoints, AWS PrivateLink
- AWS CloudTrail, AWS Lake Formation
- Amazon Macie, AWS KMS
- SageMaker Clarify

---

### Domain 4 – Operational Efficiency & Optimization (12%)

**Task Statements:**
- Choose between on-demand, provisioned throughput, and batch inference
- Implement intelligent prompt routing for cost optimization
- Use prompt caching to reduce token costs
- Optimize vector search performance and embedding dimensionality
- Implement cross-region inference for resiliency and capacity
- Monitor costs with Model Invocation Logging and cost allocation tags

**Key Services:**
- Amazon Bedrock Batch Inference
- Bedrock Provisioned Throughput
- Bedrock Intelligent Prompt Routing
- Bedrock Prompt Caching
- AWS Cost Explorer, Cost Allocation Tags
- Amazon CloudWatch, AWS X-Ray

---

### Domain 5 – Testing, Validation & Troubleshooting (12%)

**Task Statements:**
- Configure Bedrock model evaluation jobs (automated and human-based)
- Implement RAG-specific evaluation (retrieval precision, context relevance, faithfulness)
- Use regression testing frameworks to prevent quality degradation
- Debug Bedrock Agents using traces
- Diagnose hallucinations with contextual grounding checks
- Implement CI/CD quality gates for GenAI deployments

**Key Services:**
- Amazon Bedrock Model Evaluations
- Amazon Bedrock Agent Traces
- Amazon SageMaker Clarify (bias)
- Amazon CloudWatch, AWS X-Ray
- AWS CodePipeline, AWS CodeDeploy
- Amazon A2I (Augmented AI)

---

## 🧠 Core Architectural Mindset

The exam tests **judgment and architectural reasoning**, NOT memorization. For every question, think:

1. **Least operational overhead** → Prefer managed services over custom code
2. **Least custom development** → Native AWS integrations > Lambda glue code
3. **Defense-in-depth** → Layer multiple security controls
4. **Root cause, not symptoms** → Fix data quality at source, not model behavior
5. **Separation of concerns** → Security at AWS layer, not application layer

---

## 🎓 Exam Strategy Framework

### The ELIMINATE Method

For scenario questions (which dominate the exam):

1. **E**liminate solutions using wrong service category entirely
2. **L**imit candidates to those that address the CORE requirement
3. **I**dentify "least overhead" / "most managed" clue words
4. **M**atch infrastructure needs (latency-sensitive → real-time, not batch)
5. **I**nspect security requirements (must be private → VPC endpoint required)
6. **N**arrow by operational model (no code changes → AppConfig, not Lambda env vars)
7. **A**nalyze trade-offs explicitly stated in the scenario
8. **T**est against all constraints simultaneously (not just one)
9. **E**xecute: Choose the option that satisfies ALL constraints

---

### Common Trap Patterns

| Trap | What AWS Tests |
|------|---------------|
| "Just fine-tune the model" | Fine-tuning is overkill; use RAG + metadata filters |
| "Increase Lambda memory" | Does not affect model inference latency |
| "Use CloudWatch metrics alone" | Metrics show WHAT; X-Ray/Traces show WHY |
| "Add a retry loop" | Circuit Breaker pattern is better for resilience |
| "Increase chunk size to reduce costs" | Larger chunks reduce vector count but hurt precision |
| "Use response hashing for RAG testing" | LLMs are non-deterministic; hashes will always differ |
| "Route everything to Guardrails" | Balance safety AND utility; don't block legitimate queries |
| "S3 policies for RAG security" | Apply row-level security at retrieval time, not storage |

---

### Time Management

- **130 minutes / 75 questions = ~1 min 44 sec per question**
- Target: First pass in 90 minutes, use remaining 40 minutes to review flagged questions
- Flag questions you're unsure about; never leave blank
- Multi-answer questions: read how many answers are required (usually 2-3)

---

## 📌 Recommended Study Sequence

```
Week 1: Domains 1 & 2 (RAG + Agents) → Hands-on Bedrock
Week 2: Domain 3 (Security) → IAM, Guardrails, VPC Endpoints
Week 3: Domains 4 & 5 (Optimization + Testing) → Cost patterns, Evaluation
Week 4: Mixed practice questions + Scenario analysis
```

---

## 🔗 Official Resources

| Resource | Purpose |
|----------|---------|
| [Exam Guide PDF](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/) | Authoritative domain definitions |
| [AWS Skill Builder](https://explore.skillbuilder.aws/) | Official practice questions |
| [AWS Well-Architected GenAI Lens](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/) | Architectural best practices (key exam source) |
| [Bedrock User Guide](https://docs.aws.amazon.com/bedrock/) | Service-level deep dives |
| [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/) | API-level details |

---

*Last Updated: May 2026 | Based on AIP-C01 Exam Guide*
