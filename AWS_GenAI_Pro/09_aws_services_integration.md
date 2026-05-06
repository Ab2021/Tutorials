# AWS Services Integration Reference for GenAI
## AIP-C01 – Cross-Domain Service Patterns

---

## 🗺️ Complete AWS GenAI Service Map

```
DATA LAYER                    COMPUTE LAYER               AI/ML LAYER
─────────────────────────────────────────────────────────────────────────
Amazon S3                     AWS Lambda                  Amazon Bedrock
Amazon DynamoDB               Amazon ECS                    ├─ Models
Amazon Aurora                 AWS Fargate                   ├─ Knowledge Bases
Amazon OpenSearch             Amazon EC2                    ├─ Agents
Amazon Neptune                AWS Batch                     ├─ Guardrails
                                                            ├─ Flows
INTEGRATION LAYER             ORCHESTRATION               ├─ Evaluations
─────────────────────────────────────────────────────────├─ Prompt Mgmt
API Gateway (REST/WS)         AWS Step Functions          └─ AgentCore
AWS AppSync                   Amazon EventBridge        
Amazon SQS/SNS                AWS CodePipeline          SECURITY LAYER
AWS Amplify                   AWS CodeDeploy            ─────────────────
                              AppConfig                  AWS IAM
ANALYTICS LAYER                                          AWS KMS
─────────────────────         OBSERVABILITY              AWS WAF
AWS Glue                      ─────────────────────      AWS CloudTrail
Amazon Athena                 AWS X-Ray                  VPC/PrivateLink
Amazon Kinesis                Amazon CloudWatch          AWS Lake Formation
Amazon QuickSight             AWS CloudTrail             Amazon Macie
                                                          IAM Identity Center
EDGE LAYER
─────────────────
AWS CloudFront
AWS Wavelength
AWS Outposts
```

---

## 🔗 Service Decision Matrix

### Choosing the Right Data Store

| Requirement | Best Service | Why |
|-------------|-------------|-----|
| Vector search + metadata filtering | OpenSearch Serverless | ANN search + rich filter DSL |
| Vector + relational data | Aurora PostgreSQL + pgvector | SQL queries + vector similarity |
| Graph relationships + vectors | Neptune Analytics | Graph traversal + HNSW index |
| Ultra-low cost vector storage | Amazon S3 Vectors | Object-based; less query flexibility |
| Session/conversation state | DynamoDB | Key-value, fast, managed |
| Document metadata + lineage | Glue Data Catalog | Schema registry, data lineage |
| Model versioning + cards | SageMaker Model Registry | Model lifecycle, documentation |

---

### Choosing the Right Integration Pattern

| Requirement | Best Pattern | Service |
|-------------|-------------|---------|
| Streaming responses | WebSocket + streaming API | API Gateway WebSocket + ConverseStream |
| Long-running async tasks (>30s) | Async queue | SQS + Lambda |
| Multi-step workflow with conditionals | State machine | Step Functions |
| Event-driven document ingestion | Event notification | S3 Events + Lambda + StartIngestionJob |
| Configuration without code changes | External config | AWS AppConfig |
| GraphQL with React streaming | Amplify AI Kit | AWS Amplify + AppSync |

---

### Choosing the Right Observability Tool

| What You Need | Use | Rationale |
|---------------|-----|-----------|
| End-to-end distributed tracing | X-Ray | Trace spans across Lambda, API GW, Bedrock |
| Log all Bedrock API calls | CloudTrail | Control plane and data plane auditing |
| Log prompts/responses for debugging | Model Invocation Logging | Payload-level visibility |
| Real-time metrics and alarms | CloudWatch | Latency, token counts, error rates |
| Query audit logs with SQL | Athena + S3 | CloudTrail logs → Athena queries |
| Agent reasoning visibility | Agent Traces | Step-by-step reasoning chain |

---

## 🔄 Key AWS Service Deep Dives

### AWS Step Functions – Exam Favorite ⭐

**Use Cases in GenAI:**

```
1. Multi-step LLM pipelines with conditional logic
2. Human-in-the-loop approval (Task Tokens)
3. Circuit Breaker for service outages
4. Agentic reasoning with explicit audit trail
5. Security test suite automation
```

**State Types:**

| State | Description | GenAI Use |
|-------|-------------|-----------|
| **Task** | Invoke Lambda, Bedrock, etc. | FM calls, tool execution |
| **Wait** | Pause for duration or timestamp | Rate limiting between calls |
| **Choice** | Branch based on conditions | Route by query complexity |
| **Parallel** | Execute branches simultaneously | Fan-out to multiple agents |
| **Map** | Process array items | Batch processing with state |
| **Pass** | Transform data | Reshape FM output |
| **Succeed/Fail** | Terminal states | End workflow |

**Task Token Pattern (Human-in-the-loop):**
```json
{
  "Type": "Task",
  "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken",
  "Parameters": {
    "QueueUrl": "APPROVAL_QUEUE_URL",
    "MessageBody": {
      "taskToken.$": "$$.Task.Token",
      "aiOutput.$": "$.aiGeneratedContent",
      "riskScore.$": "$.riskScore"
    }
  },
  "HeartbeatSeconds": 86400,  // 24-hour timeout if no human action
  "Next": "ExecuteApprovedAction"
}
```

---

### Amazon SageMaker – Exam Relevant Services

| SageMaker Service | Purpose | Exam Context |
|------------------|---------|-------------|
| **Model Registry** | Version, tag, and manage models | Model governance, lineage |
| **Model Cards** | Structured model documentation | Intended use, limitations, training data |
| **Clarify** | Bias detection, explainability | Fairness evaluation, feature importance |
| **Model Monitor** | Continuous production monitoring | Detect drift, bias in real-time |
| **SageMaker Endpoints** | Host custom ML models | When Bedrock can't host the model |

**Model Card Example Structure:**
```json
{
  "modelId": "sentiment-classifier-v2.1",
  "intendedUses": "Customer feedback sentiment classification",
  "businessPurpose": "Route negative feedback for immediate response",
  "trainingDataset": "s3://data/sentiment-labeled-v2",
  "evaluationResults": {"f1": 0.94, "precision": 0.92, "recall": 0.96},
  "limitations": "English-language reviews only; may underperform on sarcasm",
  "ethicalConsiderations": "Do not use for employment decisions",
  "caveats": "Performance degrades on reviews < 10 words"
}
```

---

### Amazon Comprehend in GenAI Pipelines

**Primary Uses:**

```python
comprehend = boto3.client('comprehend')

# 1. PII detection before Bedrock (pre-processing layer)
pii_response = comprehend.detect_pii_entities(
    Text=user_input,
    LanguageCode='en'
)

# 2. Text normalization for noisy transcripts
entities = comprehend.detect_entities(Text=raw_transcript, LanguageCode='en')

# 3. Language detection for multilingual routing  
lang = comprehend.detect_dominant_language(Text=user_message)

# 4. Sentiment for feedback collection triggers
sentiment = comprehend.detect_sentiment(Text=response, LanguageCode='en')
if sentiment['Sentiment'] == 'NEGATIVE':
    trigger_a2i_review()
```

---

### Amazon Textract vs. Bedrock Data Automation

```
Use Textract when:
- Extracting structured data from FORMS (key-value pairs)
- Extracting data from TABLES
- Simple OCR with layout awareness
- Invoice processing (vendor, amount, line items)
- Form processing (checkboxes, signature fields)

Use Bedrock Data Automation (BDA) when:
- Complex document understanding with SEMANTIC context
- Converting PDFs/images to structured JSON for Knowledge Base ingestion
- Documents with complex layouts (multi-column, mixed content)
- Audio/video/image understanding for knowledge ingestion
```

**Textract for Structured Extraction:**
```python
textract = boto3.client('textract')

response = textract.analyze_document(
    Document={'S3Object': {'Bucket': 'invoices', 'Name': 'invoice.pdf'}},
    FeatureTypes=['FORMS', 'TABLES']
)

# Extract key-value pairs from forms
for block in response['Blocks']:
    if block['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in block['EntityTypes']:
        key_text = get_text(block, response)
        value_text = get_value(block, response)
```

---

### Amazon Kinesis for Real-Time Data Ingestion

```python
# Streaming real-time events into Bedrock Knowledge Base
kinesis = boto3.client('kinesis')

# Producer: Send real-time product events
kinesis.put_record(
    StreamName='product-updates',
    Data=json.dumps({
        "product_id": "SKU-12345",
        "name": "Updated Product Name",
        "price": 29.99,
        "timestamp": "2025-01-15T10:30:00Z"
    }),
    PartitionKey="product-catalog"
)

# Consumer (Lambda) → processes records → updates KB
# Kinesis Data Firehose → S3 → S3 Event → Lambda → StartIngestionJob
```

---

### AWS Lake Formation for Fine-Grained Access

**Exam Use Case: Multi-Account Cross-Database Access**

```
Centralized Data Lake Account (storage)
    └── S3 Bucket: sensitive-training-data
    └── AWS Glue Catalog:
            └── Database: customer_records
                    └── Table: transactions (columns: user_id, amount, timestamp, SSN)

Application Account (compute)
    └── Lambda calls Athena
    └── Lake Formation grants: 
            user_id, amount, timestamp ← ✅ Allowed for Lambda role
            SSN ← ❌ Blocked by LF column-level grant
```

```python
# Lake Formation column-level grant
lakeformation.grant_permissions(
    Principal={"DataLakePrincipalIdentifier": "arn:aws:iam::APP_ACCOUNT:role/LambdaRole"},
    Resource={
        "TableWithColumns": {
            "DatabaseName": "customer_records",
            "Name": "transactions",
            "ColumnNames": ["user_id", "amount", "timestamp"]  # Exclude SSN
        }
    },
    Permissions=["SELECT"],
    PermissionsWithGrantOption=[]
)
```

> **Exam Pattern (Q8 from samples):** "Fine-grained column-level access across AWS accounts" → **Lake Formation LF-tag-based access control** with cross-account grants

---

## 🌊 Streaming Architecture Patterns

### Pattern 1: WebSocket for Real-Time Chat

```
React Frontend
    ↕ WebSocket connection
[API Gateway WebSocket API]
    → Lambda (onMessage)
         → Bedrock InvokeModelWithResponseStream
         → Lambda posts chunks back via API Gateway @connections
    ← tokens stream back to client
```

### Pattern 2: Server-Sent Events via HTTP Streaming

```
Client → HTTP GET /stream
[Lambda Function URL with response streaming]
    → Bedrock ConverseStream
    → Write chunks to response stream
Client receives tokens as server-sent events
```

### Pattern 3: Amplify AI Kit for React + GraphQL

```
React App with @aws-amplify/ui-react-ai
    → AppSync GraphQL subscription
         → Lambda resolver
              → Bedrock RetrieveAndGenerate
         ← Streaming response via AppSync subscription
    ← Character-by-character rendering in React
```

> **Exam Pattern (Q3 from samples):** "AWS Amplify + AppSync + Bedrock with timeout issues" → **Use AWS Amplify AI Kit** (not increase Lambda timeout, not SQS)

---

## 📱 Amazon Q Business

**Purpose:** Enterprise AI assistant with native enterprise content connectors.

```
Corporate Data Sources:
    ├── Amazon S3
    ├── Microsoft SharePoint  
    ├── Salesforce
    ├── Confluence
    ├── ServiceNow
    └── Many more (50+ native connectors)
         ↓
[Amazon Q Business]
    ├── Respects existing IAM / ACL permissions
    ├── IAM Identity Center integration
    └── No custom connector code needed
         ↓
[Enterprise Chat Interface]
```

**Key Differentiator:** Respects existing document-level permissions WITHOUT custom code.

> **Exam Pattern:** "Enterprise assistant over S3, SharePoint, and Salesforce respecting existing permissions with minimal development" → **Amazon Q Business with native connectors + Identity Center**

---

## 🏢 Enterprise GenAI Architecture – Complete Reference

### Production RAG System

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SECURITY PERIMETER                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                         VPC                                  │  │
│  │                                                              │  │
│  │  [AWS WAF] → [API Gateway] → [Lambda (Provisioned)]         │  │
│  │                                     │                        │  │
│  │              [Comprehend PII] ←─────┤                        │  │
│  │                     │               │                        │  │
│  │              [Bedrock Guardrails]   │                        │  │
│  │                     │               │                        │  │
│  │  [Interface VPC Endpoint] ──────────┘                        │  │
│  │         │                                                    │  │
│  │  [Amazon Bedrock] ←── IAM Role                               │  │
│  │         │                                                    │  │
│  │  [Knowledge Base] → [OpenSearch Serverless]                  │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  OBSERVABILITY: CloudTrail + Model Invocation Logs + X-Ray          │
│  GOVERNANCE: Lake Formation + Glue Catalog + SageMaker Model Cards  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Service Selection Practice Questions

**Q1:** A company needs to process real-time product catalog updates and make them searchable in their GenAI assistant within 2 minutes. Which pipeline is correct?

- A. Amazon Kinesis → S3 → Glue ETL → OpenSearch  
- B. Amazon Kinesis → Kinesis Firehose → S3 → S3 Event → Lambda → StartIngestionJob  
- C. DynamoDB Streams → Lambda → Bedrock Knowledge Base  
- D. EventBridge scheduled rule → Lambda → StartIngestionJob every 2 minutes  

**Answer: B** – Kinesis captures real-time events → Firehose buffers to S3 → S3 event triggers Lambda → Lambda calls StartIngestionJob for near-real-time KB updates.

---

**Q2:** A legal company needs to query 5 TB of unstructured legal documents stored in S3 across multiple accounts, with column-level restrictions to protect confidential client names. Which combination is correct?

- A. S3 bucket policies + IAM roles for cross-account access  
- B. AWS Lake Formation with LF-tag-based column grants + AWS Glue catalog for cross-account  
- C. Amazon Macie to classify and restrict access to sensitive columns  
- D. AWS Config rules to enforce column-level access policies  

**Answer: B** – Lake Formation provides fine-grained column-level grants; Glue catalog enables cross-account data discovery.

---

**Q3:** A company's React app uses AppSync and Bedrock but users experience timeouts on complex queries. What should be changed?

- A. Increase the Lambda resolver timeout to 15 minutes  
- B. Change RequestResponse to Event invocation type in Lambda  
- C. Use AWS Amplify AI Kit to implement streaming responses through GraphQL  
- D. Add SQS between AppSync and Lambda  

**Answer: C** – Amplify AI Kit enables streaming responses through GraphQL, eliminating timeout issues without increasing timeout values.

---

*Next: [10_practice_questions_bank.md](./10_practice_questions_bank.md)*
