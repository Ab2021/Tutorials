# Azure AI Services — Brit Insurance Interview Prep

> [!NOTE]
> **Where this file sits.** Long-form **deep-dive** companion to `14-industry-intel-lead-ai-interview-research.md` (§ Azure) and `05-system-design-round-playbook.md`. Harvested in full from the earlier `br_ins/` research set so that none of that material is lost, with one factual correction applied (below).
>
> **⚠️ Correction applied to §8.** The original text asserted that *Brit uses Cytora in production*. That is **not verified** — see `11-insurance-lloyds-domain-primer.md` §7.1. The JD names Cytora as a **comparable, buy-side product**, not as a current Brit deployment. §8 has been rewritten to match. Read that section as *market context and a design pattern*, never as Brit's stack.

> **Target Role**: Lead AI Engineer @ Brit Insurance  
> **Core Question**: *"Which components of Azure AI have you worked on?"*  
> **JD Keywords**: Copilot Studio · Power Platform · Azure AI Services · LLMOps · "pro-code or low-code decisions"

---

## Table of Contents

1. [Azure AI Ecosystem Overview](#1-azure-ai-ecosystem-overview)
2. [Azure OpenAI Service — Deep Dive](#2-azure-openai-service--deep-dive)
3. [Azure AI Search — Deep Dive](#3-azure-ai-search--deep-dive)
4. [Azure AI Foundry](#4-azure-ai-foundry)
5. [Microsoft Copilot Studio](#5-microsoft-copilot-studio)
6. [Azure AI Document Intelligence](#6-azure-ai-document-intelligence)
7. [Azure AI Content Safety](#7-azure-ai-content-safety)
8. [Cytora — Insurance-Specific Platform](#8-cytora--insurance-specific-platform)
9. [MLOps / LLMOps on Azure](#9-mlops--llmops-on-azure)
10. [Cost Management on Azure AI](#10-cost-management-on-azure-ai)
11. [Security Architecture — Regulated Environment](#11-security-architecture--regulated-environment)
12. [Sample Interview Q&A](#12-sample-interview-qa)

---

## 1. Azure AI Ecosystem Overview

### The Full Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AZURE AI ECOSYSTEM                            │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Azure OpenAI   │    │  Azure AI Search│    │  Azure AI      │  │
│  │  Service        │    │  (Cognitive     │    │  Foundry       │  │
│  │  GPT-4o, o1,    │    │   Search)       │    │  (ML Studio +  │  │
│  │  Embeddings     │    │  Vector+Hybrid  │    │   AI Studio)   │  │
│  └────────┬────────┘    └────────┬────────┘    └───────┬────────┘  │
│           │                      │                      │           │
│  ┌────────▼─────────────────────▼──────────────────────▼────────┐  │
│  │              Azure AI Agent Service / Semantic Kernel          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Azure AI Doc   │    │  Azure AI       │    │  Azure         │  │
│  │  Intelligence   │    │  Content Safety │    │  Cognitive     │  │
│  │  (Form Recog.)  │    │  Prompt Shields │    │  Services      │  │
│  └─────────────────┘    └─────────────────┘    └────────────────┘  │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Copilot Studio │    │  Power Platform │    │  Azure ML      │  │
│  │  Low-code Agent │    │  Power Apps     │    │  Classic MLOps │  │
│  │  Builder        │    │  Power Automate │    │  Pipelines     │  │
│  └─────────────────┘    └─────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Service-by-Service Summary

| Service | What It Does | Primary Use at Brit | Key Differentiator |
|---|---|---|---|
| **Azure OpenAI** | LLM inference: GPT-4o, embeddings, DALL-E | Core reasoning engine for all GenAI features | Enterprise SLA, private networking, RBAC |
| **Azure AI Search** | Hybrid vector + keyword search with semantic reranking | Document retrieval for RAG pipelines | BM25 + HNSW + semantic reranker in one service |
| **Azure AI Foundry** | Full AI lifecycle: experiment → deploy → monitor | Prompt flow orchestration, model evaluation | Replaces both Azure ML Studio + AI Studio |
| **Azure AI Document Intelligence** | Extract structured data from unstructured docs | Parse insurance submissions, invoices, policy docs | Pre-built + custom models; handles complex layouts |
| **Azure AI Content Safety** | Detect harmful content, prompt injections | Guard rails on all user-facing AI endpoints | Sub-100ms latency at p99 |
| **Azure Cognitive Services** | Vision, Speech, Language, Translator (umbrella) | Ancillary capabilities (OCR, translation) | Many now migrated under AI Foundry |
| **Azure Machine Learning** | Classical MLOps: train, register, deploy models | Risk scoring models, fraud detection | MLflow integration, compute clusters |
| **Copilot Studio** | Low-code chatbot + agent builder | Internal helpdesk, broker-facing FAQ bots | No-code fallback when pro-code is overkill |
| **Power Platform** | Workflow automation, Power Apps, Power BI | Business process automation for ops teams | Non-developer-friendly; Dataverse backed |
| **Azure AI Agent Service** | Managed orchestration: tools + threads + memory | Multi-agent orchestration patterns | Stateful, durable agent execution |

---

## 2. Azure OpenAI Service — Deep Dive

### 2.1 Model Landscape

```python
# Models available in Azure OpenAI (2024-2025)
MODELS = {
    "gpt-4o":           {"context": 128_000, "purpose": "Best all-round, multimodal"},
    "gpt-4o-mini":      {"context": 128_000, "purpose": "Fast + cheap, good for classification"},
    "gpt-4-turbo":      {"context": 128_000, "purpose": "Predecessor to 4o, still widely used"},
    "gpt-4":            {"context":   8_192, "purpose": "Legacy; use 4o instead"},
    "o1-preview":       {"context": 128_000, "purpose": "Chain-of-thought reasoning tasks"},
    "o3-mini":          {"context": 128_000, "purpose": "Fast reasoning, cost-effective"},
    "text-embedding-3-large": {"dims": 3072, "purpose": "High-quality embeddings for RAG"},
    "text-embedding-3-small": {"dims": 1536, "purpose": "Cheaper, still very good"},
    "text-embedding-ada-002":  {"dims": 1536, "purpose": "Legacy; migrate to v3"},
}
```

### 2.2 Deployment Modes

| Mode | Billing | Best For | Latency |
|---|---|---|---|
| **Standard (Pay-as-you-go)** | Per 1K tokens | Dev/test, variable load | Variable (shared compute) |
| **Provisioned Throughput (PTU)** | Fixed hourly cost | Production, predictable load, SLA | Consistent, low |
| **Batch API** | ~50% cheaper | Offline processing, bulk jobs | Hours (async) |

> [!TIP] **PTU Decision Rule**: If your monthly token spend on a model exceeds ~$10K, PTUs are usually more cost-effective. Each PTU unit provides a guaranteed tokens-per-minute capacity.

```python
# Calculating PTU capacity needed
# 1 PTU ≈ 1,500 input TPM + 500 output TPM for GPT-4o (varies)
# If you need 60,000 TPM → need ~40 PTUs
required_tpm = 60_000
input_tpm_per_ptu = 1500
ptu_needed = required_tpm / input_tpm_per_ptu  # = 40 PTUs
```

### 2.3 Handling 429 Errors (Rate Limiting) — Production Pattern

> [!IMPORTANT] 429 errors are the #1 production issue with Azure OpenAI. You MUST handle them correctly. Never use a fixed sleep — always respect the `retry-after-ms` header.

```python
import time
import random
import logging
from typing import Any, Optional
from openai import AzureOpenAI, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """
    Production-grade Azure OpenAI client with:
    - Exponential backoff respecting Retry-After header
    - Jitter to prevent thundering-herd
    - Circuit breaker awareness
    - Comprehensive logging
    """

    def __init__(self, endpoint: str, deployment: str, api_version: str = "2024-08-01-preview"):
        # Keyless auth: uses DefaultAzureCredential (Managed Identity)
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        self.deployment = deployment

    def chat_with_retry(
        self,
        messages: list[dict],
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **kwargs: Any
    ) -> Any:
        """
        Chat completion with exponential backoff + jitter.
        Respects Azure's retry-after-ms header when present.
        """
        attempt = 0
        while attempt <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    **kwargs
                )
                return response

            except RateLimitError as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Max retries exceeded for rate limit", extra={"attempts": attempt})
                    raise

                # Respect Retry-After-Ms header if present (Azure sets this)
                retry_after_ms = self._extract_retry_after_ms(e)
                if retry_after_ms:
                    wait_s = retry_after_ms / 1000.0
                    logger.warning(f"Rate limited. Respecting Retry-After-Ms: {wait_s:.1f}s")
                else:
                    # Exponential backoff with full jitter
                    wait_s = min(base_delay * (2 ** attempt), max_delay)
                    wait_s = random.uniform(0, wait_s)  # full jitter
                    logger.warning(f"Rate limited. Backoff: {wait_s:.1f}s (attempt {attempt})")

                time.sleep(wait_s)

            except APIStatusError as e:
                # 5xx errors: also retry with backoff
                if e.status_code >= 500:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    wait_s = min(base_delay * (2 ** attempt), max_delay)
                    wait_s = random.uniform(0, wait_s)
                    logger.warning(f"Server error {e.status_code}. Backoff: {wait_s:.1f}s")
                    time.sleep(wait_s)
                else:
                    raise  # 4xx (not 429) — don't retry

    def _extract_retry_after_ms(self, error: RateLimitError) -> Optional[int]:
        """Extract Retry-After-Ms from error response headers."""
        try:
            headers = error.response.headers
            if "retry-after-ms" in headers:
                return int(headers["retry-after-ms"])
            if "retry-after" in headers:
                return int(headers["retry-after"]) * 1000
        except Exception:
            pass
        return None
```

### 2.4 Keyless Authentication with Managed Identity

> [!IMPORTANT] **Never use API keys in production code.** Use Managed Identity with `DefaultAzureCredential`. This is the FCA/Lloyd's-compliant approach — no secrets in code, no rotation risk.

```python
# app_config.py — Infrastructure-level auth, not application-level secrets
from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    AzureCliCredential,
)
from openai import AzureOpenAI

def create_openai_client(endpoint: str, api_version: str) -> AzureOpenAI:
    """
    Creates Azure OpenAI client using Managed Identity.
    
    Resolution chain:
    1. AZURE_CLIENT_ID env var (User-Assigned MI in production)
    2. System-assigned MI (fallback in Azure-hosted environments)
    3. Azure CLI (fallback for local dev)
    """
    credential = ChainedTokenCredential(
        ManagedIdentityCredential(),  # Works in AKS, App Service, Azure Functions
        AzureCliCredential(),          # Works in local dev (az login)
    )
    
    from azure.identity import get_bearer_token_provider
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

# Terraform: Grant Cognitive Services User role to the app's managed identity
# resource "azurerm_role_assignment" "aoai_user" {
#   scope                = azurerm_cognitive_account.aoai.id
#   role_definition_name = "Cognitive Services OpenAI User"
#   principal_id         = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
# }
```

### 2.5 Private Networking

```
┌─────────────────────────────────────────────────────────────┐
│  Azure Virtual Network (10.0.0.0/16)                        │
│                                                             │
│  ┌─────────────────┐     Private     ┌──────────────────┐  │
│  │   AKS Cluster   │────Endpoint────▶│  Azure OpenAI    │  │
│  │   (app subnet)  │   10.0.2.4      │  No public IP    │  │
│  │   10.0.1.0/24   │                 │  disabled        │  │
│  └─────────────────┘                 └──────────────────┘  │
│                                                             │
│  ┌─────────────────┐                 ┌──────────────────┐  │
│  │  Private DNS    │                 │  Azure AI Search │  │
│  │  Zone           │                 │  Private EP      │  │
│  │  privatelink    │                 │  10.0.2.5        │  │
│  │  .openai.azure  │                 └──────────────────┘  │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Azure CLI: Create private endpoint for Azure OpenAI
az network private-endpoint create \
  --name pe-aoai-prod \
  --resource-group rg-brit-ai-prod \
  --vnet-name vnet-brit-prod \
  --subnet snet-private-endpoints \
  --private-connection-resource-id $(az cognitiveservices account show \
      --name aoai-brit-prod \
      --resource-group rg-brit-ai-prod \
      --query id -o tsv) \
  --group-id account \
  --connection-name plsc-aoai-prod

# Disable public network access (CRITICAL for compliance)
az cognitiveservices account update \
  --name aoai-brit-prod \
  --resource-group rg-brit-ai-prod \
  --custom-domain aoai-brit-prod \
  --public-network-access Disabled
```

### 2.6 Prompt Caching — Maximising Cache Hits

> [!TIP] Azure OpenAI caches prompt prefixes automatically. Structure prompts to put stable content (system prompt, few-shot examples) FIRST, and variable content (user query) LAST. Cache hit = ~50% cost reduction on input tokens.

```python
# ✅ GOOD: Stable prefix → gets cached across calls
def build_cached_prompt(user_query: str, document_context: str) -> list[dict]:
    return [
        {
            "role": "system",
            # STABLE: Put large system prompts here — they get cached
            "content": INSURANCE_SYSTEM_PROMPT  # e.g. 2000-token domain prompt
        },
        {
            "role": "user",
            # STABLE prefix of user turn (e.g., few-shot examples)
            "content": f"""You are analyzing insurance documents.

Examples of good analysis:
{FEW_SHOT_EXAMPLES}  # Also stable → also cached

---
DOCUMENT CONTEXT:
{document_context}

QUESTION: {user_query}"""
        }
    ]

# ✅ Token counting before sending (avoid overage)
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken (exact match for GPT-4o family)."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += len(enc.encode(msg.get("content", "")))
        total += len(enc.encode(msg.get("role", "")))
    total += 2  # assistant reply priming
    return total
```

---

## 3. Azure AI Search — Deep Dive

### 3.1 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AZURE AI SEARCH                                │
│                                                                     │
│  Data Sources        Indexers           Skills             Index   │
│  ┌──────────┐    ┌───────────┐    ┌────────────────┐    ┌───────┐  │
│  │ Blob     │───▶│ Blob      │───▶│ OCR            │───▶│ Fields│  │
│  │ Storage  │    │ Indexer   │    │ Key Phrase     │    │ + Vec │  │
│  └──────────┘    └───────────┘    │ Entity Recog.  │    │ store │  │
│  ┌──────────┐    ┌───────────┐    │ Custom AML     │    └───────┘  │
│  │ SQL DB   │───▶│ SQL       │    │ Embedding      │               │
│  └──────────┘    │ Indexer   │    │ (Azure OpenAI) │               │
│  ┌──────────┐    └───────────┘    └────────────────┘               │
│  │ CosmosDB │                                                       │
│  └──────────┘     Push API (direct document upload also supported)  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Vector Search: HNSW vs Exhaustive KNN

| Algorithm | Type | Recall | Speed | Use Case |
|---|---|---|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Approximate Nearest Neighbor | ~95-99% | Very fast O(log n) | Production, large indexes |
| **Exhaustive KNN** | Exact Nearest Neighbor | 100% | Slow O(n) | Small indexes, ground-truth eval |

```python
# Index schema with vector field
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, HnswParameters,
    VectorSearchProfile, SemanticConfiguration, SemanticSearch,
    SemanticPrioritizedFields, SemanticField
)

def create_insurance_index(client, index_name: str) -> SearchIndex:
    """
    Design an index for insurance document RAG.
    Supports: keyword search, vector search, semantic reranking.
    """
    fields = [
        SearchField(name="id",           type=SearchFieldDataType.String, key=True),
        SearchField(name="doc_type",     type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(name="insured_name", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="policy_ref",   type=SearchFieldDataType.String, filterable=True),
        SearchField(name="content",      type=SearchFieldDataType.String, searchable=True),
        SearchField(name="chunk_index",  type=SearchFieldDataType.Int32,  filterable=True, sortable=True),
        SearchField(name="source_file",  type=SearchFieldDataType.String, filterable=True),
        SearchField(name="lob",          type=SearchFieldDataType.String, filterable=True, facetable=True),
        # Row-level security: broker/underwriter access control
        SearchField(name="allowed_groups", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True),
        # Vector field (1536 dims for text-embedding-3-small)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    m=4,              # Connections per layer (4-64; higher = better recall, more memory)
                    ef_construction=400,  # Build-time search width (higher = better quality index)
                    ef_search=500,        # Query-time search width
                    metric="cosine",
                )
            )
        ],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-config")]
    )

    semantic_config = SemanticConfiguration(
        name="insurance-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="doc_type"),
                             SemanticField(field_name="insured_name")],
        )
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config])
    )
```

### 3.3 Hybrid Search: BM25 + Vector + Semantic Reranking

> [!NOTE] **Hybrid search consistently outperforms pure vector or pure BM25** for domain-specific corpora like insurance documents. Semantic reranking adds ~10-15% relevance uplift at the cost of additional latency (~200ms).

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, QueryType

def hybrid_search_insurance_docs(
    search_client: SearchClient,
    query: str,
    query_embedding: list[float],
    lob_filter: str | None = None,
    user_groups: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Full hybrid search:
      1. BM25 keyword matching on 'content' field
      2. Vector similarity on 'content_vector' field (HNSW)
      3. Semantic reranking on combined results
      4. Security filter for row-level access control
    """
    # Build OData filter
    filters = []
    if lob_filter:
        filters.append(f"lob eq '{lob_filter}'")
    if user_groups:
        # Row-level security: only return docs accessible to this user
        group_filter = " or ".join(
            [f"allowed_groups/any(g: g eq '{g}')" for g in user_groups]
        )
        filters.append(f"({group_filter})")
    odata_filter = " and ".join(filters) if filters else None

    results = search_client.search(
        search_text=query,                       # BM25 keyword
        vector_queries=[
            VectorizableTextQuery(               # Vector search
                text=query,                      # auto-vectorizes if integrated vectorization
                k_nearest_neighbors=top_k * 3,  # over-fetch for reranker
                fields="content_vector",
            )
        ],
        query_type=QueryType.SEMANTIC,           # Enable semantic reranking
        semantic_configuration_name="insurance-semantic",
        query_caption="extractive",              # Return highlighted excerpts
        query_answer="extractive",               # Return direct answers
        filter=odata_filter,
        top=top_k,
        select=["id", "content", "doc_type", "source_file", "chunk_index", "insured_name"],
    )

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "doc_type": r["doc_type"],
            "source": r["source_file"],
            "score": r["@search.reranker_score"],  # Semantic reranker score (0-4)
            "caption": r.get("@search.captions", [{}])[0].get("text", ""),
        }
        for r in results
    ]
```

### 3.4 Chunking Strategies for Insurance Documents

```python
from dataclasses import dataclass
from enum import Enum

class ChunkStrategy(Enum):
    FIXED_SIZE = "fixed"          # Simple, fast; loses context at boundaries
    SENTENCE    = "sentence"       # Better coherence
    PARAGRAPH   = "paragraph"      # Good for dense policy docs
    SEMANTIC    = "semantic"       # Best recall, expensive
    HIERARCHICAL = "hierarchical"  # Summary chunk + detail chunks (best for RAG)

@dataclass
class ChunkConfig:
    strategy: ChunkStrategy
    chunk_size: int     # tokens
    overlap: int        # tokens (20% overlap recommended)

# For insurance docs: use hierarchical chunking
# - Parent chunk: full section (512 tokens) → stored for context
# - Child chunk: paragraph (128 tokens) → used for retrieval
INSURANCE_CHUNK_CONFIG = ChunkConfig(
    strategy=ChunkStrategy.HIERARCHICAL,
    chunk_size=512,
    overlap=64,
)

def chunk_insurance_document(text: str, config: ChunkConfig) -> list[dict]:
    """
    Hierarchical chunking:
    - Embed small child chunks for precision retrieval
    - Return parent chunk for answer synthesis (more context)
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Split into sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_tokens = []
    current_text = []
    chunk_idx = 0

    for sentence in sentences:
        s_tokens = enc.encode(sentence)
        if len(current_tokens) + len(s_tokens) > config.chunk_size:
            # Save current chunk
            chunks.append({
                "chunk_index": chunk_idx,
                "content": " ".join(current_text),
                "token_count": len(current_tokens),
            })
            # Overlap: keep last N tokens worth of sentences
            overlap_tokens = 0
            overlap_sentences = []
            for prev in reversed(current_text):
                pt = len(enc.encode(prev))
                if overlap_tokens + pt > config.overlap:
                    break
                overlap_tokens += pt
                overlap_sentences.insert(0, prev)
            current_text = overlap_sentences + [sentence]
            current_tokens = enc.encode(" ".join(current_text))
            chunk_idx += 1
        else:
            current_text.append(sentence)
            current_tokens.extend(s_tokens)

    if current_text:
        chunks.append({
            "chunk_index": chunk_idx,
            "content": " ".join(current_text),
            "token_count": len(current_tokens),
        })

    return chunks
```

### 3.5 Multi-Tenant Isolation Patterns

```python
# Pattern 1: Index-per-tenant (strong isolation, higher cost)
def get_tenant_index_name(tenant_id: str) -> str:
    return f"insurance-docs-{tenant_id}"  # e.g., "insurance-docs-brit-london"

# Pattern 2: Single index with security filter (cost-effective, correct for Brit)
# Each document tagged with allowed_groups; queries always include group filter
# This is what Brit likely uses: same index, broker-level row-security

# Pattern 3: Semantic filter partitions (Azure AI Search partitioning)
# scoringProfile with document boosts per partition key
```

---

## 4. Azure AI Foundry

### 4.1 What Is AI Foundry?

Azure AI Foundry (released 2024) **unified** three previously separate services:
- **Azure Machine Learning Studio** (classical MLOps)
- **Azure AI Studio** (GenAI experimentation)
- **Prompt Flow** (orchestration, now first-class)

```
AI Foundry Hub (centralized governance)
├── Project A (RAG for Submissions)
│   ├── Connections (Azure OpenAI, AI Search, Blob)
│   ├── Prompt Flows
│   ├── Evaluations
│   └── Deployments
├── Project B (Document Intelligence Pipeline)
└── Project C (Risk Scoring Models)
```

### 4.2 Prompt Flow — Orchestration

```yaml
# promptflow/insurance_rag.yaml
inputs:
  query:
    type: string
  lob:
    type: string
    default: "property"

nodes:
  - name: embed_query
    type: llm
    source:
      type: code
      path: embed_query.py
    inputs:
      query: ${inputs.query}

  - name: retrieve_docs
    type: python
    source:
      path: search_docs.py
    inputs:
      query_embedding: ${embed_query.output}
      lob: ${inputs.lob}

  - name: generate_answer
    type: llm
    source:
      type: code
      path: generate_answer.jinja2
    inputs:
      query: ${inputs.query}
      context: ${retrieve_docs.output}
    connection: azure_openai_connection
    api: chat

outputs:
  answer:
    type: string
    reference: ${generate_answer.output}
  sources:
    type: list
    reference: ${retrieve_docs.output}
```

### 4.3 Built-in Evaluation Metrics

```python
# Evaluation with AI Foundry SDK
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
)

def evaluate_rag_pipeline(test_dataset: list[dict], model_config: dict):
    """
    Evaluate a RAG pipeline on a golden test set.
    Returns: DataFrame with per-metric scores.
    """
    evaluators = {
        "groundedness":  GroundednessEvaluator(model_config=model_config),
        "relevance":     RelevanceEvaluator(model_config=model_config),
        "coherence":     CoherenceEvaluator(model_config=model_config),
        "fluency":       FluencyEvaluator(model_config=model_config),
    }

    results = []
    for sample in test_dataset:
        row = {"question": sample["question"]}
        for name, evaluator in evaluators.items():
            score = evaluator(
                question=sample["question"],
                answer=sample["generated_answer"],
                context=sample["retrieved_context"],
            )
            row[name] = score[name]
        results.append(row)

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.describe())
    return df

# Eval gate: fail CI if groundedness drops below threshold
EVAL_THRESHOLDS = {
    "groundedness": 4.0,   # Scale: 1-5
    "relevance":    3.5,
    "coherence":    4.0,
}
```

### 4.4 Fine-Tuning: When and When Not To

| Situation | Fine-tune? | Alternative |
|---|---|---|
| Model doesn't know your domain terms | ✅ Yes | Or: RAG with glossary |
| Need consistent output format | ❌ No | Structured output / JSON mode |
| Latency is critical (shorter prompts) | ✅ Consider | Prompt compression |
| Limited labelled data (<100 examples) | ❌ No | Few-shot in prompt |
| Need to reduce hallucination | ❌ No | RAG + grounding evaluation |
| Unique stylistic voice/tone | ✅ Consider | RLHF / DPO fine-tune |

> [!WARNING] Fine-tuning costs are significant ($3-8 per 1M training tokens). Always validate that fine-tuning outperforms RAG + good prompting before committing.

---

## 5. Microsoft Copilot Studio

### 5.1 What It Is

Copilot Studio is Microsoft's **low-code platform** for building conversational AI agents. It sits on top of Power Platform and Azure AI, and is designed for **citizen developers** and **business analysts** — not ML engineers.

```
User Channel (Teams, Web, WhatsApp)
        │
        ▼
┌─────────────────────────────────────┐
│       Copilot Studio Bot            │
│  ┌──────────────────────────────┐   │
│  │  Topics (dialog flows)       │   │
│  │  ┌────────────────────────┐  │   │
│  │  │ Trigger phrases        │  │   │
│  │  │ Conditions             │  │   │
│  │  │ Actions (call API)     │  │   │
│  │  │ Generative AI fallback │  │   │
│  │  └────────────────────────┘  │   │
│  └──────────────────────────────┘   │
│  Custom Actions ──────────────────▶ Azure Function / Power Automate │
│  Knowledge Sources ───────────────▶ SharePoint / Websites / AI Search │
└─────────────────────────────────────┘
```

### 5.2 When Brit Uses Copilot Studio vs Pro-Code

| Scenario | Copilot Studio | Pro-Code (Python + Azure) |
|---|---|---|
| Broker FAQ chatbot | ✅ Perfect fit | Overkill |
| HR policy Q&A | ✅ SharePoint integration built-in | Overkill |
| Complex submission analysis | ❌ Too limited | ✅ Required |
| Multi-step underwriting workflow | ❌ Use Power Automate instead | ✅ For complex logic |
| Customer-facing with SLA | ❌ Limited monitoring | ✅ Full observability |
| Rapid prototype in 1 week | ✅ Fast iteration | Slower setup |

### 5.3 Custom Actions and Connectors

```json
// Custom connector definition for Cytora API
{
  "swagger": "2.0",
  "info": { "title": "Cytora Risk API", "version": "1.0" },
  "host": "api.cytora.com",
  "basePath": "/v2",
  "paths": {
    "/submissions/triage": {
      "post": {
        "operationId": "TriageSubmission",
        "summary": "Score and triage an insurance submission",
        "parameters": [{
          "name": "body",
          "in": "body",
          "schema": {
            "type": "object",
            "properties": {
              "submission_text": { "type": "string" },
              "lob": { "type": "string" },
              "insured_name": { "type": "string" }
            }
          }
        }],
        "responses": {
          "200": {
            "schema": {
              "properties": {
                "risk_score": { "type": "number" },
                "appetite_match": { "type": "boolean" },
                "triage_decision": { "type": "string" }
              }
            }
          }
        }
      }
    }
  }
}
```

### 5.4 Limitations — When NOT to Use Copilot Studio

> [!CAUTION] Copilot Studio has hard limits that make it unsuitable for production insurance AI:
> - **No streaming responses** — users see full answer at once (poor UX for long answers)
> - **Limited custom middleware** — can't inject complex retrieval logic
> - **Observability gaps** — Application Insights integration is partial
> - **No fine-grained RBAC** on individual topics
> - **Dataverse dependency** — all state goes through Dataverse (latency, cost)
> - **4000 character limit** on single message responses

### 5.5 Governance Features

```yaml
# Copilot Studio governance checklist for regulated environments
governance:
  - content_moderation: Azure AI Content Safety integration (built-in toggle)
  - authentication:     Entra ID SSO required for internal bots
  - audit_logging:      Power Platform admin center audit logs
  - data_residency:     Set to UK South / UK West for FCA compliance
  - dlp_policies:       Power Platform DLP to block external connector calls
  - version_control:    Export solution as ZIP → commit to Azure DevOps
```

---

## 6. Azure AI Document Intelligence

### 6.1 Pre-Built vs Custom Models

```
┌─────────────────────────────────────────────────────────────────────┐
│              AZURE AI DOCUMENT INTELLIGENCE                         │
│                                                                     │
│  Pre-Built Models              Custom Models                        │
│  ┌──────────────────────┐     ┌──────────────────────────────────┐  │
│  │ • Invoice            │     │ • Custom Template Model          │  │
│  │ • Receipt            │     │   (fixed layout, e.g., Lloyd's   │  │
│  │ • Business Card      │     │    slip format)                  │  │
│  │ • Contract           │     │ • Custom Neural Model            │  │
│  │ • Health Insurance   │     │   (variable layout,              │  │
│  │   Card               │     │    e.g., broker submissions)     │  │
│  │ • US Tax W2, 1099    │     │ • Composed Model                 │  │
│  │ • Bank Statement     │     │   (multiple models merged)       │  │
│  │ • General Document   │     └──────────────────────────────────┘  │
│  └──────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Insurance Document Extraction

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.identity import DefaultAzureCredential
from dataclasses import dataclass, field

@dataclass
class InsuranceSubmission:
    insured_name: str = ""
    broker_name: str = ""
    lob: str = ""              # Line of Business
    sum_insured: float = 0.0
    policy_period: str = ""
    risk_location: str = ""
    extracted_fields: dict = field(default_factory=dict)
    confidence: float = 0.0

class InsuranceDocumentParser:
    """
    Parse insurance submissions using Azure AI Document Intelligence.
    Uses custom model trained on Lloyd's slip format.
    """

    MODEL_ID = "insurance-submission-v3"  # Custom trained model

    def __init__(self, endpoint: str):
        credential = DefaultAzureCredential()
        self.client = DocumentAnalysisClient(endpoint=endpoint, credential=credential)

    def parse_submission(self, document_url: str) -> InsuranceSubmission:
        """Analyze document and extract key insurance fields."""
        poller = self.client.begin_analyze_document_from_url(
            model_id=self.MODEL_ID,
            document_url=document_url,
        )
        result = poller.result()
        submission = InsuranceSubmission()

        if result.documents:
            doc = result.documents[0]
            fields = doc.fields

            # Extract with confidence thresholding
            def extract_field(name: str, min_confidence: float = 0.80) -> str:
                f = fields.get(name)
                if f and f.confidence >= min_confidence:
                    return f.value or ""
                return ""  # Below threshold → flag for human review

            submission.insured_name  = extract_field("InsuredName")
            submission.broker_name   = extract_field("BrokerName")
            submission.lob           = extract_field("LineOfBusiness")
            submission.risk_location = extract_field("RiskLocation")
            submission.confidence    = doc.confidence

            # Sum insured needs special parsing (currency)
            sum_field = fields.get("SumInsured")
            if sum_field and sum_field.value:
                try:
                    raw = sum_field.value.replace(",", "").replace("£", "").replace("$", "")
                    submission.sum_insured = float(raw)
                except ValueError:
                    pass

        return submission

    def parse_with_layout(self, document_bytes: bytes) -> dict:
        """
        Layout analysis: extract tables, key-value pairs, paragraphs.
        Useful for complex policy documents where structure matters.
        """
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=document_bytes,
        )
        result = poller.result()

        output = {"tables": [], "kv_pairs": [], "paragraphs": []}

        for table in result.tables:
            rows = {}
            for cell in table.cells:
                if cell.row_index not in rows:
                    rows[cell.row_index] = {}
                rows[cell.row_index][cell.column_index] = cell.content
            output["tables"].append(rows)

        for kv in result.key_value_pairs:
            if kv.key and kv.value:
                output["kv_pairs"].append({
                    "key": kv.key.content,
                    "value": kv.value.content,
                    "confidence": kv.confidence,
                })

        return output
```

### 6.3 Integration with RAG Pipelines

```python
# Pipeline: PDF Submission → Doc Intelligence → Chunk → Embed → AI Search
async def process_submission_pipeline(blob_url: str, tenant_id: str):
    """
    End-to-end document processing pipeline.
    Called from Azure Function on Blob Storage trigger.
    """
    # 1. Extract structured fields + raw text
    parser = InsuranceDocumentParser(endpoint=DOC_INTEL_ENDPOINT)
    submission = parser.parse_submission(blob_url)
    layout = parser.parse_with_layout(await download_blob(blob_url))

    # 2. Combine structured + unstructured text
    full_text = build_enriched_text(submission, layout)

    # 3. Chunk the document
    chunks = chunk_insurance_document(full_text, INSURANCE_CHUNK_CONFIG)

    # 4. Embed each chunk
    embeddings = await batch_embed(
        texts=[c["content"] for c in chunks],
        model="text-embedding-3-small"
    )

    # 5. Upload to AI Search
    documents = [
        {
            "id": f"{submission.insured_name}-{i}",
            "content": chunk["content"],
            "content_vector": embeddings[i],
            "doc_type": "submission",
            "insured_name": submission.insured_name,
            "lob": submission.lob,
            "chunk_index": i,
            "source_file": blob_url,
            "allowed_groups": [f"underwriter-{submission.lob}", "admin"],
        }
        for i, chunk in enumerate(chunks)
    ]
    search_client.upload_documents(documents=documents)
```

---

## 7. Azure AI Content Safety

### 7.1 What It Protects Against

```
User Input ──▶ [Content Safety API] ──▶ Allow/Block Decision ──▶ Azure OpenAI
                        │
                ┌───────▼────────────────────────────────┐
                │  Checks performed (in parallel):        │
                │  • Prompt Shield (injection detection)  │
                │  • Hate/Violence/Sexual/Self-harm       │
                │  • Custom blocklist matching            │
                │  • Indirect injection (in documents)    │
                └────────────────────────────────────────┘

LLM Response ─▶ [Content Safety API] ──▶ Allow/Block Decision ──▶ User
                        │
                ┌───────▼─────────────────────────────────┐
                │  • Groundedness check (is answer in doc?)│
                │  • Hate/Violence/Sexual/Self-harm         │
                │  • Protected material detection           │
                └─────────────────────────────────────────┘
```

### 7.2 Implementation Without Unacceptable Latency

> [!TIP] Run Content Safety checks **in parallel** with other prep work (embedding the query, fetching user context) to absorb its latency (~50-100ms). Never run it sequentially before every other step.

```python
import asyncio
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.identity.aio import DefaultAzureCredential

class ContentSafetyGuard:
    """
    Async Content Safety checks run in parallel with query processing.
    Target: add <20ms net latency to the critical path.
    """

    SEVERITY_THRESHOLD = 2  # 0=safe, 2=low, 4=medium, 6=high (block at >= 4)

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def check_input(self, user_input: str) -> dict:
        """Check user input for harmful content AND prompt injection."""
        async with ContentSafetyClient(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential()
        ) as client:
            # Run both checks concurrently
            harm_task       = client.analyze_text(AnalyzeTextOptions(
                text=user_input,
                categories=[
                    TextCategory.HATE, TextCategory.VIOLENCE,
                    TextCategory.SEXUAL, TextCategory.SELF_HARM,
                ],
            ))
            shield_task     = client.detect_text_prompt_injection({"userPrompt": user_input})

            harm_result, shield_result = await asyncio.gather(harm_task, shield_task)

        # Evaluate results
        blocked = False
        reasons = []
        for cat in harm_result.categories_analysis:
            if cat.severity >= self.SEVERITY_THRESHOLD:
                blocked = True
                reasons.append(f"{cat.category}: severity={cat.severity}")

        if shield_result.user_prompt_attack_detected:
            blocked = True
            reasons.append("prompt_injection_detected")

        return {"blocked": blocked, "reasons": reasons}

    async def check_groundedness(
        self, answer: str, grounding_sources: list[str]
    ) -> dict:
        """Verify LLM answer is grounded in retrieved documents."""
        async with ContentSafetyClient(
            endpoint=self.endpoint, credential=DefaultAzureCredential()
        ) as client:
            result = await client.detect_groundedness(
                domain="Generic",
                task="QA",
                qna={"query": "insurance question", "answer": answer},
                grounding_sources=grounding_sources,
                reasoning=True,  # Get explanation (adds ~100ms)
            )
        return {
            "ungrounded": result.ungrounded,
            "score": result.ungroundedness_score,
            "reason": result.reasoning,
        }

# Usage pattern: parallel safety check
async def safe_rag_query(query: str, user_groups: list[str]) -> str:
    guard = ContentSafetyGuard(CONTENT_SAFETY_ENDPOINT)

    # Run safety check and query embedding simultaneously
    safety_task   = guard.check_input(query)
    embedding_task = embed_query_async(query)
    safety, embedding = await asyncio.gather(safety_task, embedding_task)

    if safety["blocked"]:
        return f"Request blocked: {', '.join(safety['reasons'])}"

    # Proceed with search + generation...
    docs    = hybrid_search_insurance_docs(search_client, query, embedding, ...)
    answer  = await generate_answer(query, docs)

    # Post-generation groundedness check
    ground_result = await guard.check_groundedness(answer, [d["content"] for d in docs])
    if ground_result["ungrounded"]:
        return "Unable to provide a grounded answer from available documents."

    return answer
```

### 7.3 Insurance-Specific Safety Requirements

| Requirement | Implementation |
|---|---|
| No PII leakage across tenants | Security filters in AI Search + content safety scan on output |
| No market-sensitive data exposure | Custom blocklist with competitor names, pricing keywords |
| Regulatory compliance in responses | Groundedness detection + disclaimer injection |
| Audit trail of all rejections | Log to Azure Monitor with structured fields |

---

## 8. Cytora — Insurance-Specific Platform (market context, **not** a Brit deployment)

> [!WARNING]
> **Read this before the section.** Research found **no evidence of a Brit–Cytora deployment** (`11-insurance-lloyds-domain-primer.md` §7.1). Brit's public broker-platform news is **Acturis**; Cytora's public partnership news is with **Polaris (2019)**, before its acquisition by **Applied Systems in 2025**. Everything below is *how the class of product works and how you would integrate with one* — not a statement about Brit's estate. **Never claim Brit uses Cytora in the room.**

### 8.1 What Cytora Is

Cytora is an **AI-native risk processing platform** purpose-built for commercial insurance. It sits between brokers/insureds and underwriting teams, automating the submission triage and risk evaluation workflow.

```
Broker Submission                 Underwriter Decision
(Email/PDF/API)                   (Bind/Decline/Referral)
       │                                    ▲
       ▼                                    │
┌──────────────────────────────────────────┴──────────────────────┐
│                         CYTORA                                    │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ Ingestion        │  │ Risk Intelligence│  │ Workflow       │  │
│  │ • Email parsing  │  │ • Risk scoring   │  │ • Route to UW  │  │
│  │ • PDF extraction │  │ • Appetite match │  │ • SLA tracking │  │
│  │ • API intake     │  │ • Data enrichment│  │ • Audit trail  │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
│           │                    │                      │           │
│           └────────────────────▼──────────────────────┘          │
│                    Integration Layer                              │
│              (REST API / Webhooks / Azure Event Grid)            │
└─────────────────────────────────────────────────────────────────┘
         │                                         │
         ▼                                         ▼
   Azure AI Services                       Policy Admin System
   (OpenAI, Doc Intel,                     (Guidewire, Duck Creek,
    AI Search)                              Sequel, etc.)
```

### 8.2 Key Cytora Features

| Feature | What It Does | Illustrative use case |
|---|---|---|
| **Submission Ingestion** | Parse emails, attachments, API calls into structured data | Capture Lloyd's broker submissions from multiple channels |
| **Risk Scoring** | ML model scores risk across >100 features | Pre-qualify risks before UW time allocation |
| **Appetite Matching** | Match submission against configurable UW appetite rules | Auto-decline out-of-appetite risks instantly |
| **Triage & Routing** | Route to correct UW team with priority | Property → Property UW; Marine → Marine UW |
| **Data Enrichment** | Augment with external data (flood maps, crime stats, Companies House) | Better informed risk assessment |
| **Workflow Automation** | Orchestrate back-and-forth referral workflows | Reduce touchpoints per submission |

### 8.3 Cytora + Azure AI Integration Pattern

```python
# Cytora webhook handler → Azure Function → Azure AI enrichment
import azure.functions as func
import json

async def process_cytora_event(req: func.HttpRequest) -> func.HttpResponse:
    """
    Cytora fires this webhook when a new submission is ingested.
    We enrich it using Azure AI before returning to Cytora's workflow.
    """
    payload = req.get_json()
    submission_id = payload["submissionId"]
    raw_text      = payload["extractedText"]   # Cytora's initial extraction
    
    # 1. Enhanced extraction via Azure AI Document Intelligence
    parser = InsuranceDocumentParser(DOC_INTEL_ENDPOINT)
    submission = parser.parse_submission(payload["documentUrl"])

    # 2. RAG-based risk context retrieval
    search_client = get_search_client()
    similar_risks = hybrid_search_insurance_docs(
        search_client=search_client,
        query=f"{submission.lob} {submission.risk_location} similar risks",
        query_embedding=await embed_query_async(raw_text),
        lob_filter=submission.lob,
        top_k=3,
    )

    # 3. AI-generated underwriting summary
    uw_summary = await generate_uw_summary(submission, similar_risks)

    # 4. Push enrichment back to Cytora via API
    cytora_response = {
        "submissionId": submission_id,
        "enrichments": {
            "ai_uw_summary":    uw_summary,
            "similar_risk_refs": [r["id"] for r in similar_risks],
            "extracted_fields":  {
                "insured_name":  submission.insured_name,
                "sum_insured":   submission.sum_insured,
                "lob":           submission.lob,
            },
            "confidence_score":  submission.confidence,
        }
    }
    return func.HttpResponse(json.dumps(cytora_response), mimetype="application/json")
```

### 8.4 If the panel raises Cytora — what to say

> [!IMPORTANT]
> The original version of this section asserted *"At Brit, Cytora handles the front door of the underwriting process."* **That claim was unverified and has been removed.** Do not repeat it.

The JD names Cytora alongside Copilot Studio as the **class of comparable buy-side tooling**. That reads as *the option to be evaluated*, not a system already in service. Answer as a **design pattern**, not as a deployment fact:

1. A submission-intake platform of this class owns the **front door** — broker email/ACORD/API intake, extraction, classification, appetite pre-screen.
2. Your AI layer's job is the **seam**, not the whole pipeline: enrich via Document Intelligence, ground via AI Search, summarise via Azure OpenAI, and **push enrichment back through the vendor's API** (the webhook pattern in §8.3).
3. The design rules that make that seam safe — and the ones worth saying out loud:
   - **Idempotency keys** on every write-back, so a retried webhook does not double-post an enrichment
   - A **`correlationId`** carried through the vendor hop, your function, and your audit record
   - A **hard timeout** on your enrichment call, so your latency never becomes the vendor's SLA breach
   - An **audit record** of exactly what the model added versus what the vendor supplied — the trace is the audit evidence
   - **Human review** on any AI-influenced decline or referral; automated rejection is the highest-risk action in the flow

**The line to say:**

> *"The JD names Cytora as the comparable class of tooling, so I'd read that as the build-or-buy decision to be evaluated rather than something already in service. Either way the shape is the same — the AI layer sits at the seam, doing enrichment and producing the evidence trail, and the vendor keeps ownership of intake."*

---

## 9. MLOps / LLMOps on Azure

### 9.1 MLOps vs LLMOps: Key Differences

| Dimension | Classical MLOps | LLMOps |
|---|---|---|
| **What changes** | Model weights, features | Prompts, retrieval config, model version |
| **Testing unit** | Model accuracy metrics | Groundedness, relevance, safety scores |
| **Versioning artifact** | Model + training data | Prompt template + index config + model |
| **Drift type** | Data drift, model drift | Prompt drift, retrieval drift, model API changes |
| **Compute** | Training clusters | Token budget management |
| **Latency SLA** | Batch acceptable | Usually p99 < 3s for user-facing |

### 9.2 CI/CD Pipeline for LLM Systems

```yaml
# .github/workflows/llm-deploy.yml
name: LLM RAG Pipeline — Build, Evaluate, Deploy

on:
  push:
    branches: [main]
    paths:
      - 'prompts/**'
      - 'src/rag/**'
      - 'config/search_config.yaml'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with: { python-version: '3.11' }

      - name: Install deps
        run: pip install -r requirements-eval.txt

      - name: Run evaluation against golden dataset
        env:
          AZURE_OPENAI_ENDPOINT:   ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_SEARCH_ENDPOINT:   ${{ secrets.AZURE_SEARCH_ENDPOINT }}
        run: |
          python scripts/evaluate_rag.py \
            --dataset tests/golden_qa_dataset.jsonl \
            --output   eval_results.json \
            --fail-on  groundedness<4.0 \
            --fail-on  relevance<3.5

      - name: Upload evaluation results
        uses: actions/upload-artifact@v3
        with:
          name: eval-results
          path: eval_results.json

  deploy:
    needs: evaluate
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy prompt flow to AI Foundry
        run: |
          pfazure run create \
            --flow ./promptflow/insurance_rag \
            --data ./tests/golden_qa_dataset.jsonl \
            --subscription ${{ secrets.AZURE_SUBSCRIPTION_ID }} \
            --resource-group rg-brit-ai-prod \
            --workspace-name ai-foundry-brit-prod

      - name: Update AI Search index config
        run: |
          python scripts/deploy_index_config.py \
            --config config/search_config.yaml \
            --env production
```

### 9.3 Three Types of Drift in LLM Systems

```python
# drift_monitor.py — Detect prompt, retrieval, and model drift

class DriftMonitor:
    """Monitor three layers of LLM system drift."""

    def detect_prompt_drift(self, eval_history: list[dict]) -> dict:
        """
        Prompt drift: evaluation scores degrade without any code changes.
        Cause: LLM API model updates (Azure rolls out updated model versions).
        Detection: scheduled eval runs against frozen golden dataset.
        """
        scores = [r["groundedness"] for r in eval_history]
        trend  = scores[-1] - scores[0]  # Negative = degrading
        return {"type": "prompt_drift", "trend": trend, "alert": trend < -0.3}

    def detect_retrieval_drift(self, search_eval_history: list[dict]) -> dict:
        """
        Retrieval drift: retrieval quality degrades as new documents are added.
        Cause: Index grows stale, chunk quality varies, embedding model changes.
        Detection: Track MRR/NDCG on retrieval eval set over time.
        """
        mrr_scores = [r["mrr"] for r in search_eval_history]
        trend = mrr_scores[-1] - mrr_scores[0]
        return {"type": "retrieval_drift", "trend": trend, "alert": trend < -0.05}

    def detect_data_drift(self, query_logs: list[str], reference_queries: list[str]) -> dict:
        """
        Data drift: user query distribution shifts from baseline.
        Cause: New product launches, regulatory changes, seasonal patterns.
        Detection: Embedding-based distribution shift (MMD or cosine similarity).
        """
        # Compare centroid of recent queries vs reference queries
        recent_emb   = embed_batch(query_logs[-1000:])
        reference_emb = embed_batch(reference_queries)
        
        import numpy as np
        recent_centroid   = np.mean(recent_emb, axis=0)
        reference_centroid = np.mean(reference_emb, axis=0)
        shift = float(np.dot(recent_centroid, reference_centroid) /
                      (np.linalg.norm(recent_centroid) * np.linalg.norm(reference_centroid)))
        return {"type": "data_drift", "cosine_similarity": shift, "alert": shift < 0.90}
```

### 9.4 Prompt Versioning

```python
# prompts/v1.2.3/system_prompt.jinja2
# Prompts are versioned alongside code in Git
# Version tag format: major.minor.patch (same semver as software)

# prompts/registry.py
PROMPT_REGISTRY = {
    "rag_system_prompt": {
        "v1.0.0": "prompts/v1.0.0/system_prompt.jinja2",
        "v1.1.0": "prompts/v1.1.0/system_prompt.jinja2",
        "v1.2.3": "prompts/v1.2.3/system_prompt.jinja2",  # current production
        "v2.0.0-beta": "prompts/v2.0.0-beta/system_prompt.jinja2",
    }
}

# config/environments/production.yaml
prompts:
  rag_system_prompt: v1.2.3
  extraction_prompt: v2.1.0
  safety_prompt:     v1.0.5

search:
  index_name:         insurance-docs-prod
  index_version:      "20241115"

model:
  completion_model:   gpt-4o-2024-08-06   # Pin exact model version
  embedding_model:    text-embedding-3-small-1
```

### 9.5 Monitoring Stack

```python
# monitoring/ai_monitor.py — Application Insights custom events for AI

from opencensus.ext.azure.log_exporter import AzureLogHandler
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=APP_INSIGHTS_CONN_STR))

def log_rag_trace(
    query: str,
    retrieved_docs: list[dict],
    answer: str,
    groundedness_score: float,
    latency_ms: int,
    model: str,
    tokens_used: int,
):
    """Log every RAG call as a structured event to Application Insights."""
    logger.info(
        "rag_query_completed",
        extra={
            "custom_dimensions": {
                "query_length":         len(query),
                "retrieved_doc_count":  len(retrieved_docs),
                "top_doc_score":        retrieved_docs[0]["score"] if retrieved_docs else 0,
                "groundedness_score":   groundedness_score,
                "latency_ms":           latency_ms,
                "model":                model,
                "tokens_used":          tokens_used,
                "timestamp":            datetime.utcnow().isoformat(),
            }
        }
    )

# KQL query in Azure Monitor to alert on groundedness degradation:
# AzureDiagnostics
# | where Category == "ai_custom_events"
# | extend groundedness = todouble(custom_dimensions.groundedness_score)
# | summarize avg_groundedness = avg(groundedness) by bin(TimeGenerated, 1h)
# | where avg_groundedness < 3.5
```

---

## 10. Cost Management on Azure AI

### 10.1 Token Cost Model

```python
# cost_calculator.py — Real-time cost estimation before sending requests

# Azure OpenAI pricing (approximate, verify at azure.com/pricing)
PRICING = {
    "gpt-4o": {
        "input":  0.0025 / 1000,   # $0.0025 per 1K input tokens
        "output": 0.010  / 1000,   # $0.010  per 1K output tokens (4x input!)
        "cached_input": 0.00125 / 1000,  # 50% discount for cached prefix
    },
    "gpt-4o-mini": {
        "input":  0.000150 / 1000,
        "output": 0.000600 / 1000,
    },
    "text-embedding-3-small": {
        "input": 0.00002 / 1000,
        "output": 0.0,
    },
}

def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> dict:
    p = PRICING[model]
    uncached_input = input_tokens - cached_tokens
    cost = (
        uncached_input * p["input"]
        + cached_tokens * p.get("cached_input", p["input"])
        + output_tokens * p["output"]
    )
    return {
        "model":          model,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
        "cached_tokens":  cached_tokens,
        "estimated_usd":  cost,
    }

# Key insight: output tokens are 4x more expensive than input!
# Optimize by: shorter system prompts, fewer retrieved chunks, concise answers
```

### 10.2 Cost Optimization Strategies

```python
# Strategy 1: Model Routing — use cheapest model that can do the job
def route_to_model(query: str, task_type: str) -> str:
    """Route tasks to the most cost-effective capable model."""
    ROUTING = {
        "classification":     "gpt-4o-mini",  # 16x cheaper than gpt-4o
        "extraction":         "gpt-4o-mini",
        "summarization":      "gpt-4o-mini",
        "complex_reasoning":  "gpt-4o",        # Only when needed
        "document_analysis":  "gpt-4o",
    }
    return ROUTING.get(task_type, "gpt-4o-mini")

# Strategy 2: Semantic Caching — avoid redundant LLM calls
class SemanticCache:
    """Cache LLM responses using embedding similarity."""
    SIMILARITY_THRESHOLD = 0.92  # Cosine similarity above this → cache hit

    def __init__(self, redis_client, embedding_model):
        self.redis = redis_client
        self.embed = embedding_model

    def get(self, query: str) -> str | None:
        query_emb = self.embed(query)
        # HNSW lookup in Redis Stack / Azure Cache for Redis Enterprise
        result = self.redis.execute_command("FT.SEARCH", "idx:cache",
            f"*=>[KNN 1 @embedding $vec AS score]",
            "PARAMS", "2", "vec", query_emb.tobytes(),
            "RETURN", "2", "response", "score",
            "DIALECT", "2"
        )
        if result and float(result[-1][-1]) >= self.SIMILARITY_THRESHOLD:
            return result[-1][-2]   # Cache hit!
        return None

# Strategy 3: Batching — use Batch API for non-realtime workloads
# 50% cost reduction for async document processing
def submit_batch_job(documents: list[str]) -> str:
    """Submit bulk document analysis as a batch job (up to 24hr turnaround)."""
    requests = [
        {"custom_id": f"doc-{i}", "method": "POST", "url": "/chat/completions",
         "body": {"model": "gpt-4o", "messages": [
             {"role": "user", "content": f"Analyze this insurance document: {doc}"}
         ]}}
        for i, doc in enumerate(documents)
    ]
    # Write to JSONL and submit to Batch API
    # Returns batch_id for status polling
    ...

# Strategy 4: Output token control
RESPONSE_FORMAT_TIPS = {
    "Use max_tokens": "Cap output — don't let model ramble",
    "Use JSON mode": "Structured output is more token-efficient than prose",
    "Compress context": "Summarize retrieved docs before including in prompt",
    "Truncate history": "Don't include entire conversation history in every call",
}
```

### 10.3 PTU vs Consumption Break-Even Analysis

```
Monthly Token Spend Analysis:

Pay-as-you-go cost:
  Input:  50M tokens × $0.0025/1K  = $125/month
  Output: 10M tokens × $0.010/1K   = $100/month
  Total PAYg = $225/month

Provisioned Throughput (PTU):
  ~20 PTUs needed for this load
  PTU cost: 20 × $2/hour × 730 hours = $29,200/month  ← NOT cost-effective here

  Break-even point: ~$10K-15K/month PAYg spend → PTU becomes viable
  Also: PTU gives guaranteed throughput + lower latency (critical for prod)
```

---

## 11. Security Architecture — Regulated Environment

### 11.1 Zero-Trust Architecture for AI Systems

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ZERO-TRUST AI ARCHITECTURE                          │
│                                                                     │
│  Identity Layer:                                                    │
│  [Entra ID] ──▶ OIDC/OAuth2 ──▶ Every service call authenticated   │
│                                                                     │
│  Network Layer:                                                     │
│  [No public endpoints] + [Private DNS] + [VNet peering]            │
│                                                                     │
│  Data Layer:                                                        │
│  [Customer-Managed Keys] + [Double encryption] + [TLS 1.3 only]    │
│                                                                     │
│  Application Layer:                                                 │
│  [RBAC per tool] + [Managed Identity] + [No stored secrets]        │
│                                                                     │
│  Audit Layer:                                                       │
│  [Diagnostic logs] + [Azure Monitor] + [Immutable audit store]     │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 RBAC for AI Services

```python
# RBAC roles for Azure AI services (principle of least privilege)
RBAC_ASSIGNMENTS = {
    # Role: App Service / AKS Workload Identity
    "app_identity": [
        "Cognitive Services OpenAI User",    # Can call OpenAI inference
        "Search Index Data Reader",           # Can query AI Search (not create/delete)
        "Storage Blob Data Reader",           # Can read documents
        # NOT: Cognitive Services OpenAI Contributor (can't change deployments)
        # NOT: Search Service Contributor (can't modify index schema)
    ],

    # Role: CI/CD Pipeline Service Principal
    "ci_cd_identity": [
        "Cognitive Services OpenAI Contributor",  # For prompt flow deployment
        "Search Service Contributor",             # For index schema updates
        "Storage Blob Data Contributor",          # For test data upload
    ],

    # Role: Data Scientists / ML Engineers
    "data_scientist": [
        "Azure ML Data Scientist",
        "Cognitive Services OpenAI User",
        # AI Search: read-only via separate dev index
    ],

    # Role: Underwriters (end users)
    "underwriter_user": [
        # No direct Azure RBAC — access via application layer
        # Application enforces row-level security via AI Search filters
    ],
}
```

### 11.3 PII Handling in AI Systems

```python
# pii_handler.py — Detect and redact PII before sending to LLM
from azure.ai.textanalytics import TextAnalyticsClient, RecognizeEntitiesAction
from azure.identity import DefaultAzureCredential

class PIIRedactor:
    """
    Redact PII from documents before indexing or sending to LLM.
    Required for: GDPR compliance, Lloyd's market data governance.
    """

    # PII categories to always redact for insurance context
    REDACT_CATEGORIES = {
        "Person", "PhoneNumber", "Email",
        "Address", "NationalRegistrationNumber",
        "UKNationalHealthNumber", "CreditCardNumber",
        "BankAccountNumber", "PassportNumber",
    }

    def __init__(self, language_endpoint: str):
        self.client = TextAnalyticsClient(
            endpoint=language_endpoint,
            credential=DefaultAzureCredential()
        )

    def redact(self, text: str, preserve_for_audit: bool = True) -> tuple[str, dict]:
        """
        Returns (redacted_text, pii_map).
        pii_map: {placeholder → original} stored encrypted for audit trail.
        """
        result = self.client.recognize_pii_entities([text])[0]
        redacted = text
        pii_map = {}

        for entity in sorted(result.entities, key=lambda e: e.offset, reverse=True):
            if entity.category in self.REDACT_CATEGORIES:
                placeholder = f"[{entity.category.upper()}_{entity.offset}]"
                pii_map[placeholder] = entity.text
                redacted = redacted[:entity.offset] + placeholder + redacted[entity.offset + entity.length:]

        return redacted, pii_map
```

### 11.4 Audit Logging for FCA/Lloyd's Compliance

```python
# audit_logger.py — Immutable audit log for all AI interactions
import json
from datetime import datetime, UTC
from azure.storage.blob import BlobServiceClient, StandardBlobTier

class AIAuditLogger:
    """
    Write-once, read-many audit log to Azure Blob Storage with:
    - Immutability policy (WORM: Write Once Read Many)
    - Geographic replication (UK South + UK West)
    - 7-year retention (Lloyd's requirement)
    """

    def __init__(self, storage_endpoint: str, container: str = "ai-audit-logs"):
        self.client = BlobServiceClient(
            account_url=storage_endpoint,
            credential=DefaultAzureCredential()
        )
        self.container = container

    def log_interaction(
        self,
        user_id: str,
        session_id: str,
        action: str,            # "query", "document_access", "decision_support"
        input_summary: str,     # NOT the raw input if it contains PII
        output_summary: str,
        model: str,
        tokens_used: int,
        safety_flags: list[str],
    ):
        record = {
            "timestamp":     datetime.now(UTC).isoformat(),
            "user_id":       user_id,
            "session_id":    session_id,
            "action":        action,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "model":         model,
            "tokens_used":   tokens_used,
            "safety_flags":  safety_flags,
        }

        # Blob name: partitioned by date for efficient querying
        blob_name = (
            f"{datetime.now(UTC).strftime('%Y/%m/%d')}/"
            f"{session_id}/{datetime.now(UTC).timestamp()}.json"
        )
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_name
        )
        blob_client.upload_blob(
            json.dumps(record),
            overwrite=False,        # Prevent tampering
            standard_blob_tier=StandardBlobTier.COOL,  # Cost-effective storage
        )
```

### 11.5 Data Residency for UK Insurance

```bash
# Ensure all AI services deployed in UK regions only
# UK South: Primary region (London data centre)
# UK West: Secondary region (Cardiff data centre)

# Azure Policy: enforce UK region restriction
az policy assignment create \
  --name "enforce-uk-residency" \
  --policy "e56962a6-4747-49cd-b67b-bf8b01975c4f" \  # Allowed locations policy
  --params '{"listOfAllowedLocations": {"value": ["ukSouth", "ukWest"]}}' \
  --scope /subscriptions/${SUBSCRIPTION_ID}/resourceGroups/rg-brit-ai-prod

# AOAI: Set data zone to UK
az cognitiveservices account create \
  --name aoai-brit-prod \
  --resource-group rg-brit-ai-prod \
  --location uksouth \
  --kind OpenAI \
  --sku S0 \
  --custom-domain aoai-brit-prod
  # Data at rest: encrypted with CMK in UK South Key Vault
```

---

## 12. Sample Interview Q&A

---

### Q1: Which components of Azure AI have you worked on?

**Model Answer:**

> "Across production projects I've worked extensively with several Azure AI components. The core of most of my work has been **Azure OpenAI Service** — deploying GPT-4o for conversational and extraction tasks, using text-embedding-3-small for RAG pipelines, and implementing production patterns like keyless Managed Identity authentication, exponential backoff for 429 handling, and private endpoint networking.
>
> On retrieval, I've built hybrid search systems with **Azure AI Search**, combining BM25 keyword search, HNSW vector search, and semantic reranking — consistently getting better relevance than either approach alone. I've designed index schemas for document corpora including vector fields and row-level security via OData filters.
>
> For document processing, I've used **Azure AI Document Intelligence** — both pre-built models (invoices, contracts) and custom-trained models for domain-specific document layouts. These feed into RAG indexing pipelines.
>
> On the safety side, I've integrated **Azure AI Content Safety** — prompt shields for injection detection, groundedness detection on outputs, and running checks in parallel with query embedding to minimize latency impact.
>
> For the full AI lifecycle, I've used **Azure AI Foundry** for prompt flow orchestration, built-in evaluation with metrics like groundedness and relevance, and CI/CD integration with eval gates that block deployment if scores fall below threshold.
>
> Finally, at the governance and ops level, I've worked with **Copilot Studio** for low-code internal bots and integrated with **Power Platform** connectors, while reserving pro-code Python solutions for complex reasoning tasks where Copilot Studio's limitations become blockers."

---

### Q2: How do you handle 429 errors from Azure OpenAI in production?

**Model Answer:**

> "There are several layers to handling 429s correctly. First, I always respect the `retry-after-ms` response header — Azure sets this precisely, and using it avoids both under- and over-waiting. When the header isn't present, I fall back to exponential backoff with **full jitter** (randomising the wait time) to prevent thundering-herd effects when multiple instances hit the limit simultaneously.
>
> Beyond retries, the real solution is capacity management: if you're consistently hitting 429s, you need either more PTU capacity or better request throttling on the application side. I implement a token bucket rate limiter in the application layer so we never send more requests than our deployment can handle.
>
> For user-facing systems, I also return a `Retry-Later` response to the client rather than holding the connection open — streaming partial progress helps with UX. And I track 429 rate and backoff duration in Application Insights so I can set alerts and capacity plan proactively."

---

### Q3: How would you implement hybrid search in Azure AI Search?

**Model Answer:**

> "Hybrid search in Azure AI Search combines three complementary signals: BM25 keyword matching, HNSW vector similarity, and semantic reranking.
>
> The index schema needs a standard searchable text field for BM25 and a vector field (Collection of Single) with HNSW configuration specifying the embedding dimensions, M value, and ef parameters. I typically use `text-embedding-3-small` (1536 dims) for the balance of quality and cost.
>
> At query time, I send both a `search_text` parameter (triggering BM25) and a `VectorQuery` targeting the vector field. Azure Search uses **Reciprocal Rank Fusion (RRF)** to merge the two result sets before passing them to the semantic reranker.
>
> The semantic reranker is the most impactful addition — it uses a cross-encoder model to rescore the top-N results for actual semantic relevance rather than just similarity. It adds ~150-200ms latency but typically gives 10-15% relevance improvement, which is worth it for insurance document retrieval where precision matters.
>
> I also always add OData security filters to enforce row-level access — underwriters should only retrieve documents their team has access to."

---

### Q4: What's the difference between Azure OpenAI and Azure AI Foundry?

**Model Answer:**

> "Azure OpenAI is the **inference service** — it exposes the API endpoints for GPT-4o, embeddings, and other models. It handles the compute, scaling, rate limiting, and billing for model calls.
>
> Azure AI Foundry is the **lifecycle management platform** — it's where you experiment, evaluate, orchestrate, and deploy AI applications. It wraps Azure OpenAI (and other model providers) and adds: a model catalog for comparing options, Prompt Flow for visual/code workflow orchestration, built-in evaluation with metrics like groundedness and relevance, fine-tuning management, and deployment versioning.
>
> In practice: your application code calls Azure OpenAI directly for inference. Azure AI Foundry is your development and operations platform — you use it to build, test, evaluate, and CI/CD your prompts and pipelines. A useful analogy: Azure OpenAI is the database engine; AI Foundry is the application framework and IDE around it."

---

### Q5: How do you ensure data stays within a regulated perimeter?

**Model Answer:**

> "Three layers: network, identity, and policy.
>
> **Network**: All Azure AI services deployed with public network access disabled. Private endpoints in our VNet with Private DNS zones resolve service hostnames to private IPs. No data traverses the public internet — even the AOAI API call goes over Microsoft's backbone.
>
> **Identity**: Managed Identity with no API keys stored anywhere. Roles are scoped to minimum required — app identity gets `Cognitive Services OpenAI User`, not Contributor. RBAC assignments are in Terraform, versioned and reviewed.
>
> **Policy**: Azure Policy enforces that all AI services must be in UK South or UK West. Data residency is explicitly configured on each service. For regulated data, we use Customer-Managed Keys in Key Vault so even Microsoft can't access our data at rest. AI Search and AOAI both support CMK.
>
> Audit-wise: all interactions are logged to an immutable WORM-enabled Azure Blob container in UK South, replicated to UK West, with a 7-year retention policy matching Lloyd's requirements. Every interaction is logged with user ID, session ID, model used, token count, and safety flags — no raw PII in the audit log (we redact before logging)."

---

### Q6: How would you implement keyless authentication for Azure AI services?

**Model Answer:**

> "The pattern is: Managed Identity on the compute resource + RBAC role assignment on the AI service + `DefaultAzureCredential` in code.
>
> In production on AKS, I enable the OIDC issuer and Workload Identity on the cluster, create a Kubernetes service account federated with an Entra ID managed identity, and assign that identity the `Cognitive Services OpenAI User` role on the AOAI resource. The pod gets a projected service account token that `DefaultAzureCredential` exchanges for an Azure AD token automatically.
>
> In code, I use `azure.identity.DefaultAzureCredential` with `get_bearer_token_provider` to get a token provider function, then pass it to the AzureOpenAI client as `azure_ad_token_provider`. The token is cached and refreshed automatically by the SDK before expiry.
>
> For local development, `DefaultAzureCredential` falls back to `AzureCliCredential` — developers run `az login` and get access without any secrets. No API keys exist anywhere in the system — not in environment variables, not in Key Vault, not in config files. This is the approach that satisfies FCA and Lloyd's security requirements."

---

## Quick Reference: Brit Insurance Azure AI Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│              BRIT INSURANCE — AI ARCHITECTURE SUMMARY               │
│                                                                     │
│  Submission Intake (illustrative — Brit's vendor is unconfirmed):   │
│    Vendor intake → Azure Function → Doc Intelligence → AI Search    │
│    (the JD names Cytora as a comparable; see §8 warning)            │
│                                                                     │
│  Underwriting Assistant (RAG):                                      │
│    User → Content Safety → AI Search (Hybrid) → OpenAI GPT-4o     │
│         → Groundedness Check → Audit Log → User                    │
│                                                                     │
│  Internal Tooling:                                                  │
│    Copilot Studio → Power Platform → SharePoint/Dataverse          │
│                                                                     │
│  MLOps:                                                             │
│    GitHub Actions → AI Foundry Eval → Prompt Flow Deploy           │
│                                                                     │
│  Security Perimeter:                                                │
│    Private Endpoints + Managed Identity + UK Region Policy         │
│    + WORM Audit Logs + CMK Encryption                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Harvested from `br_ins/05_azure_ai_services.md` · §8 correction applied 2026-10-12 · Target: Brit Insurance Lead AI Engineer*
