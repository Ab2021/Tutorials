# AI Safety, Security & Governance
## AIP-C01 – Domain 3 (20%) – Complete Guide

---

## 🛡️ Amazon Bedrock Guardrails

Guardrails are **programmable safety controls** applied to both inputs (prompts) and outputs (responses).

### Guardrail Components

```
Input (Prompt) 
    → [Guardrail: Input Check]
    → [Foundation Model]
    → [Guardrail: Output Check]
    → Response to User
```

| Component | What It Does | Exam Scenario |
|-----------|-------------|--------------|
| **Content Filters** | Block harmful content (hate, violence, sexual, insults) | "Block inappropriate content" |
| **Denied Topics** | Block entire conversation topics | "Never discuss competitor products" |
| **Word Filters** | Block specific words/phrases | "Block competitor brand names" |
| **Sensitive Info (PII)** | Detect and mask/block personal data | "Protect PHI in medical chatbot" |
| **Grounding Check** | Verify response is based on retrieved context | "Prevent hallucinations in RAG" |
| **Contextual Grounding** | Score response against source context | "Block responses not supported by docs" |

---

## 📋 Guardrail Policies – Detailed Breakdown

### 1. Content Filters

Apply thresholds (None, Low, Medium, High) to block:

```
Categories: HATE | INSULTS | SEXUAL | VIOLENCE | MISCONDUCT | PROMPT_ATTACK
Levels: NONE | LOW | MEDIUM | HIGH
Input/Output: Independent control for each direction
```

```python
contentPolicy = {
    "filters": [
        {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
        {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}
    ]
}
```

> **Exam Trap:** Content filters → for general harmful content  
> Denied topics → for specific business-domain restrictions

---

### 2. Denied Topics ⭐

Block entire **topic areas** from both input and output.

```python
deniedTopics = [
    {
        "name": "CompetitorRecommendations",
        "definition": "Discussion of competitor products, pricing, or features",
        "examples": ["Tell me about CompetitorX", "Is CompetitorY better?"],
        "type": "DENY"
    },
    {
        "name": "SpecificFinancialAdvice",
        "definition": "Providing specific investment recommendations or guaranteed returns",
        "type": "DENY"
    }
]
```

> **Exam Pattern (Finance Company):** "Prevent AI from recommending specific stocks or guaranteed returns" → **Denied Topics Guardrail**

---

### 3. Word Filters ⭐

Block specific words or phrases. Controls both input (user) and output (model).

```python
wordPolicy = {
    "wordsConfig": [
        {"text": "CompetitorA"},
        {"text": "CompetitorB"},
        {"text": "guaranteed returns"}
    ],
    "managedWordListsConfig": [
        {"type": "PROFANITY"}  # AWS-managed profanity list
    ]
}
```

**Actions:** BLOCK (stop processing) or NONE (allow)

> **Exam Pattern:** "Block competitor names in financial assistant" → **Word Filters with BLOCK action**  
> `Answer: ADF` from SET 12 Q2: A (Denied Topics) + D (Word Filters) + F (High Grounding Score)

---

### 4. Sensitive Information Filtering (PII) ⭐

```python
sensitiveInformationPolicy = {
    "piiEntitiesConfig": [
        {"type": "NAME", "action": "ANONYMIZE"},
        {"type": "EMAIL", "action": "BLOCK"},
        {"type": "PHONE", "action": "ANONYMIZE"},
        {"type": "SSN", "action": "BLOCK"},
        {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
        {"type": "ADDRESS", "action": "ANONYMIZE"}
    ],
    "regexesConfig": [
        {
            "name": "PatientID",
            "description": "Internal patient identifier",
            "pattern": "PAT-[0-9]{8}",
            "action": "ANONYMIZE"
        }
    ]
}
```

**Actions:**
- `ANONYMIZE` – Replace with `[NAME]`, `[EMAIL]` placeholders
- `BLOCK` – Stop request entirely

> **Exam Pattern:** "Log prompts for compliance while masking PII" → **Guardrails + Model Invocation Logging** (Guardrails masks before logging)

---

### 5. Grounding Check ⭐

Evaluates whether the model's response is supported by the retrieved context.

```
Context (from KB): "Product X is available in blue and red colors only."
Response: "Product X comes in blue, red, and green colors."
          ↓ Grounding check detects "green" is NOT in context
          → BLOCKED (below grounding threshold)
```

```python
groundingPolicy = {
    "groundingThreshold": 0.7,  # 0-1; higher = stricter
    "relevanceThreshold": 0.6   # Response must be relevant to query
}
```

**Thresholds (Critical for Exam):**

| Setting | Effect | When to Use |
|---------|--------|------------|
| **High threshold** (0.8-1.0) | Strict; blocks marginally grounded responses | Medical, legal, financial accuracy required |
| **Low threshold** (0.1-0.3) | Permissive; allows creative latitude | Creative writing, brainstorming |

> **Exam Pattern (Finance Q2):** "Ensure AI assistant only makes factually grounded claims from approved guidance" → **High Grounding Score Threshold (F)**

---

## 🔒 Defense-in-Depth Architecture

**Single controls fail. Multiple layered controls provide resilience.**

```
Internet Traffic
     ↓
[AWS WAF] ← Block malicious patterns, SQL injection, known attack signatures
     ↓
[API Gateway] ← Rate limiting, authentication
     ↓
[Amazon Comprehend] ← Pre-processing: PII detection, sentiment, intent classification
     ↓
[Bedrock Guardrails] ← Content filters, denied topics, grounding checks
     ↓
[Foundation Model]
     ↓
[Bedrock Guardrails] ← Output filters (same guardrail, output side)
     ↓
[Lambda Post-Processing] ← Custom business logic validation
     ↓
[Response to User]
```

**Why Each Layer Matters:**

| Layer | Catches | Fails Against |
|-------|---------|--------------|
| WAF | Network attacks, bots | Semantic prompt injection |
| Comprehend | PII, language classification | Complex jailbreaks |
| Guardrails | Content policy violations | Custom business logic |
| Post-processing Lambda | Business rule violations | Infrastructure failures |

> **Exam Key:** Relying on ONE control = single point of failure. Always layer for comprehensive coverage.

---

## 🔑 IAM for Bedrock Access Control

### Bedrock Model Access via IAM

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSpecificModels",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1"
      ]
    }
  ]
}
```

> **Exam Key:** Access to Bedrock models = **IAM Policies on model ARNs**  
> NOT S3 bucket policies, NOT network policies alone

### Department-Level Access with IAM Identity Center

```
IAM Identity Center (SSO)
    ├── Legal Department Group → Permission Set A → Can access Claude Sonnet only
    ├── Finance Department Group → Permission Set B → Can access specific Knowledge Bases
    └── Marketing Department Group → Permission Set C → Read-only model access
```

> **Exam Pattern:** "Enforce department-level model access across organization" → **IAM Identity Center + Permission Sets with least-privilege IAM policies**

---

## 🌐 Private Network Architecture

### VPC Endpoints for Bedrock

**Why use VPC Endpoints?**
- Bedrock calls via public internet expose traffic to network risks
- VPC Interface Endpoints route traffic entirely within AWS network
- Meets compliance requirements for private connectivity

```
Lambda (private subnet)
    ↓
[Interface VPC Endpoint for Bedrock]
    ↓ (No public internet)
[Amazon Bedrock]
```

```python
# Endpoint creation (via console or CloudFormation)
# Service name: com.amazonaws.us-east-1.bedrock-runtime
# Type: Interface
# VPC: Your application VPC
# Subnets: Private subnets where Lambda runs
# Security Group: Allows HTTPS from Lambda security group
```

> **Exam Pattern:** "GenAI traffic must never traverse the public internet" → **Interface VPC Endpoints with AWS PrivateLink**

### VPC Endpoint Policy (Restrict which principals can use endpoint)

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::ACCOUNT_ID:role/BedrockLambdaRole"},
      "Action": "bedrock:InvokeModel",
      "Resource": "*"
    }
  ]
}
```

---

## 📊 Audit, Compliance & Data Lineage

### AWS CloudTrail for Bedrock

Captures every API call to Bedrock:

```json
{
  "eventSource": "bedrock.amazonaws.com",
  "eventName": "InvokeModel",
  "userIdentity": {
    "type": "AssumedRole",
    "arn": "arn:aws:sts::ACCOUNT:assumed-role/LambdaRole/function"
  },
  "requestParameters": {
    "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"
  },
  "eventTime": "2025-01-15T10:30:00Z"
}
```

> **Exam Pattern:** "Capture which IAM principal invoked which model" → **AWS CloudTrail**  
> CloudTrail = WHO made the call + WHEN + WHAT operation

### Data Lineage: Glue Data Catalog + SageMaker Model Registry

```
Training Data (S3)
    ↓
[AWS Glue Data Catalog] ← Records data source, schema, transformations
    ↓
[SageMaker Training Job]
    ↓
[SageMaker Model Registry]
    ├── Model Version 1.0 → linked to training data ARN
    │   Model Card:
    │   - Intended use: Customer sentiment classification
    │   - Training data: S3://data/sentiment-v1 (from Glue catalog)
    │   - Performance metrics: F1=0.94
    │   - Limitations: English only
    └── Model Version 2.0 → linked to training data ARN
```

> **Exam Pattern:** "Track which training data produced which model version" → **SageMaker Model Registry + AWS Glue Data Catalog**

### S3 Object Lock for Training Data Immutability

```python
# Enable compliance mode Object Lock
s3.put_object_lock_configuration(
    Bucket="training-data-bucket",
    ObjectLockConfiguration={
        "ObjectLockEnabled": "Enabled",
        "Rule": {
            "DefaultRetention": {
                "Mode": "COMPLIANCE",  # Cannot be overridden even by root
                "Days": 2555  # 7 years
            }
        }
    }
)
```

> **Exam Pattern:** "Training data in S3 must be immutable and auditable" → **S3 Object Lock (COMPLIANCE mode) + CloudTrail S3 data events**

---

## 🏥 Data Residency Requirements

### When Data Cannot Leave a Region

**Standard VPC Endpoints** secure network transit but compute still runs in AWS data centers (potentially multi-AZ within region).

**For STRICT physical data residency:**

```
Sensitive Data (must stay on-premises)
    ↓
[AWS Outposts] ← AWS infrastructure in your data center
    ↓ De-identify / anonymize data
[De-identified Data]
    ↓
[Amazon Bedrock] (regional AWS endpoint)
```

> **Exam Pattern:** "Sensitive data must NEVER leave geographic region; processing must occur locally" → **AWS Outposts + de-identification before sending to Bedrock**

---

## 🔐 Encryption

### Encryption at Rest

```
S3 (training data, logs) → S3-SSE-KMS with CMK
OpenSearch Serverless (vector store) → Encryption with CMK
DynamoDB (session state) → Encryption with CMK
```

### Encryption in Transit

```
Application → TLS 1.2+ → VPC Endpoint → Bedrock
```

### Customer-Managed Keys (CMK)

```python
# KMS key for Bedrock Knowledge Base
kms_key = kms.create_key(
    Description="Bedrock Knowledge Base Encryption Key",
    KeyPolicy=json.dumps({...})
)

# Apply to Knowledge Base
bedrock_agent.create_knowledge_base(
    serverSideEncryptionConfiguration={
        "kmsKeyArn": kms_key["KeyMetadata"]["Arn"]
    }
)
```

---

## 🌍 Responsible AI & Bias Detection

### SageMaker Clarify

Analyzes model outputs for bias and fairness:

```
Metrics computed by Clarify:
- Disparate Impact (DI): ratio of positive outcomes across groups
- Statistical Parity Difference (SPD): difference in positive rates
- Equal Opportunity Difference: difference in true positive rates
```

### SageMaker Model Monitor + Clarify (Continuous Monitoring)

```
Production Predictions
    ↓
[SageMaker Model Monitor] ← Continuously captures prediction data
    ↓
[SageMaker Clarify] ← Computes fairness metrics on captured data
    ↓
[Amazon CloudWatch] ← Metrics published
    ↓
[CloudWatch Alarms] ← Alert if bias > threshold
```

> **Exam Pattern:** "Continuous automated bias detection in production" → **SageMaker Model Monitor + SageMaker Clarify + CloudWatch Alarms**

### Amazon A2I (Augmented AI) for Human Review

```
Low-confidence or low-rated AI response
    ↓
[Amazon A2I] ← Routes to human reviewers (SMEs)
    ↓
Human annotation / correction
    ↓
[Structured feedback] → Informs prompt engineering improvements
```

**Workflow Configuration:**

```python
a2i.create_human_review_workflow(
    WorkflowDefinitionName="LowRatingReview",
    HumanLoopConfig={
        "WorkteamArn": "arn:aws:sagemaker:...",
        "TaskDescription": "Review and correct this AI response"
    },
    TriggerConditions=[{
        "ConditionType": "Sampling",
        "ConditionParameters": {"RandomSamplingPercentage": 10}
    }]
)
```

> **Exam Pattern:** "Improve prompts based on low-quality user interactions" → **API captures feedback → DynamoDB stores context → A2I routes to SMEs → Annotations inform prompt engineering**

---

## 🛡️ Security Testing for GenAI

### Adversarial Prompt Testing

```
Adversarial Test Suite:
1. Prompt injection attempts
2. Jailbreak attempts ("Ignore previous instructions...")
3. PII extraction attempts
4. Competitor mention bypass
5. Toxic content bypass
```

### Automated Security Testing with Step Functions

```
CI/CD Pipeline
    ↓
[Step Functions: Run Adversarial Test Suite]
    ├── Execute adversarial prompt 1 → Check response
    ├── Execute adversarial prompt 2 → Check response
    └── Execute adversarial prompt N → Check response
    ↓
Pass/Fail gate
    ↓ Pass
Deploy to production
```

> **Exam Pattern:** "Systematically validate security defenses before deployment" → **Adversarial prompt test suite + Step Functions automation**

---

## 📝 Practice Questions (Security & Safety)

**Q1:** A financial services company needs to ensure its AI assistant never provides specific stock recommendations, never generates content about competitors, and only makes claims grounded in approved financial documents. Which THREE guardrail configurations are needed? (Choose 3)

- A. Add high-risk conversation patterns (stock recommendations, guaranteed returns) to denied topics  
- B. Configure content filter guardrail to filter high-risk conversation patterns  
- C. Configure content filter to filter competitor names  
- D. Add competitor names as word filters with BLOCK action  
- E. Set a low grounding score threshold  
- F. Set a high grounding score threshold  

**Answer: A, D, F**
- A: Denied Topics blocks entire topic areas (stock recommendations)
- D: Word Filters specifically blocks competitor brand names
- F: High grounding threshold ensures only document-backed claims pass
- B: Content filters are for harmful content categories, not business policy topics
- E: Low threshold would ALLOW ungrounded responses (wrong direction)

---

**Q2:** A company processes sensitive patient data in Lambda functions that call Bedrock. They must ensure network traffic to Bedrock never traverses the public internet. What is required?

- A. Use a NAT Gateway for all outbound Lambda traffic  
- B. Create Interface VPC Endpoints for Bedrock Runtime in the application VPC  
- C. Deploy CloudFront in front of Bedrock API  
- D. Use VPN connection to AWS backbone network  

**Answer: B** – Interface VPC Endpoints route traffic via AWS private network. NAT Gateway still routes to public Bedrock endpoints.

---

**Q3:** A company must demonstrate which specific training datasets contributed to a specific model version for regulatory compliance. Which combination of services is correct?

- A. Amazon CloudWatch + Amazon Bedrock Model Evaluations  
- B. AWS Glue Data Catalog + SageMaker Model Registry with Model Cards  
- C. Amazon S3 versioning + AWS Config  
- D. AWS CloudTrail + Amazon Macie  

**Answer: B** – Glue Data Catalog tracks data lineage; SageMaker Model Registry + Model Cards link model versions to training data ARNs.

---

*Next: [06_prompt_engineering_optimization.md](./06_prompt_engineering_optimization.md)*
