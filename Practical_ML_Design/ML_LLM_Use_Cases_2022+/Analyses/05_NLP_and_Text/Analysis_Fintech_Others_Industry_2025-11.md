# Fintech & Specialized Industries Analysis: NLP & Text (2023-2025)

**Analysis Date**: November 2025  
**Category**: 04_NLP_and_Text  
**Industry**: Fintech, Gaming, & Specialized Sectors  
**Articles Analyzed**: 10+ (Bloomberg, Morgan Stanley, Goldman Sachs, Roblox, Unity)  
**Period Covered**: 2023-2025  
**Research Method**: Web search synthesis + Folder Content

---

## PART 1: USE CASE OVERVIEW

### 1.1 Basic Information

**Category**: NLP & Text  
**Industry**: Fintech, Banking, Gaming  
**Companies**: Bloomberg, Morgan Stanley, Goldman Sachs, Roblox, Unity  
**Years**: 2024-2025 (Primary focus)  
**Tags**: Domain-Specific LLMs, RAG, Voice Safety, Coding Assistants

**Use Cases Analyzed**:
1.  **Bloomberg**: BloombergGPT (50B Parameter Finance Model) (2023-2024)
2.  **Morgan Stanley**: AI @ Morgan Stanley Assistant (OpenAI RAG) (2024)
3.  **Goldman Sachs**: Internal Coding Assistant & "GS AI" Platform (2024)
4.  **Roblox**: Real-time Voice Safety & Toxicity Detection (2024)
5.  **Unity**: Muse Chat (Game Dev Assistant) (2024)

### 1.2 Problem Statement

**What business problem are they solving?**

1.  **Domain Jargon**: "Bullish on long-dated treasuries" means something specific. Generic GPT-4 might miss the nuance. Bloomberg needs a model that speaks "Finance".
2.  **Advisor Efficiency**: Morgan Stanley advisors spend hours digging through PDFs to answer client questions. They need instant, accurate answers.
3.  **Developer Velocity**: Goldman Sachs has millions of lines of proprietary Slang/Java code. Generic Copilot doesn't know their internal libraries.
4.  **Child Safety**: Roblox has millions of kids talking in real-time. They need to catch bullying instantly, not 5 minutes later.

**What makes this problem ML-worthy?**

-   **Latency**: Roblox voice safety must run in milliseconds to be effective.
-   **Accuracy**: In Fintech, a hallucination ("Buy stock X") is a lawsuit. RAG must be 100% grounded.
-   **Data Privacy**: Goldman Sachs code cannot leave their VPC. They need self-hosted or strictly governed models.

---

## PART 2: SYSTEM DESIGN DEEP DIVE

### 2.1 High-Level Architecture (The "Specialist" Stack)

Specialized industries build **Domain-Specific** layers on top of generic LLMs.

```mermaid
graph TD
    A[User Input] --> B[Domain Router]
    
    subgraph "Fintech (Bloomberg)"
    B --> C[BloombergGPT]
    C --> D[Financial Tokenizer]
    end
    
    subgraph "Gaming (Roblox)"
    B --> E[Audio Stream]
    E --> F[Transformer Classifier]
    F --> G[Safety Policy Engine]
    end
    
    subgraph "RAG (Morgan Stanley)"
    B --> H[Vector Search (Internal PDFs)]
    H --> I[GPT-4 (Summarization)]
    end
    
    C & G & I --> J[Final Action/Response]
```

### 2.2 Detailed Architecture: BloombergGPT (2024)

Bloomberg proved that **Domain-Specific Pre-training** beats generic scaling for niche tasks.

**The Model**:
-   **Size**: 50 Billion Parameters.
-   **Training Data**: 363 Billion tokens of financial data (News, Filings) + 345 Billion tokens of general text.
-   **Tokenizer**: Custom tokenizer trained on financial corpus to handle numbers and tickers efficiently.
-   **Result**: Outperforms GPT-3 on financial benchmarks (Sentiment, NER) while maintaining general capability.

### 2.3 Detailed Architecture: Morgan Stanley RAG (2024)

Morgan Stanley built the **"Gold Standard" for Enterprise RAG**.

**The Pipeline**:
-   **Ingestion**: OCRs and indexes 100,000+ proprietary research reports.
-   **Retrieval**: Uses OpenAI Embeddings but strictly limits context to *only* retrieved documents.
-   **Governance**: Every response comes with a "Click to Verify" link pointing to the source PDF.
-   **Adoption**: Used by 98% of advisor teams. It's not a toy; it's a core tool.

### 2.4 Detailed Architecture: Roblox Voice Safety (2024)

Roblox built **Real-Time Multimodal Safety**.

**The Stack**:
-   **Input**: Raw audio stream.
-   **Processing**:
    1.  **ASR**: Speech-to-Text (transcription).
    2.  **Audio Style**: Analyzes tone (screaming, whispering).
    3.  **Fusion**: Combines text + tone to detect "Aggression" or "Grooming".
-   **Inference**: Runs on CPU (via Ray) to save costs at massive scale (millions of concurrent streams).

---

## PART 3: MLOPS & INFRASTRUCTURE

### 3.1 Training & Serving

**Bloomberg**:
-   **Training**: Trained on AWS SageMaker using 512 A100 GPUs.
-   **Evaluation**: Created a new benchmark "FiQA" (Financial Question Answering) because standard NLP benchmarks were too easy/irrelevant.

**Unity (Muse Chat)**:
-   **Project Awareness**: The LLM context is dynamically populated with the *current project state* (Unity Version, Render Pipeline, Active Scene). It doesn't just write code; it writes code *that works in your specific project*.

### 3.2 Evaluation Metrics

| Metric | Purpose | Company |
| :--- | :--- | :--- |
| **Financial Sentiment F1** | Accuracy of market prediction | Bloomberg |
| **Source Attribution** | % of answers with valid links | Morgan Stanley |
| **False Positive Rate** | Don't ban innocent kids | Roblox |
| **Code Acceptance Rate** | % of AI code committed | Goldman Sachs |

---

## PART 4: KEY ARCHITECTURAL PATTERNS

### 4.1 Mixed-Domain Pre-training
**Used by**: Bloomberg.
-   **Concept**: Train on 50% Domain Data + 50% General Data.
-   **Why**: Pure domain models forget how to speak English. Pure general models don't know Finance. The mix is optimal.

### 4.2 Citations-First RAG
**Used by**: Morgan Stanley.
-   **Concept**: The UI highlights the *source* before the *answer*.
-   **Why**: Builds trust. Advisors need to know *who* said it (e.g., "According to the Q3 Strategy Report...").

### 4.3 Multimodal Classification
**Used by**: Roblox.
-   **Concept**: Text alone isn't enough. "I'm going to kill you" said laughing vs. screaming has different meanings.
-   **Why**: Context is crucial for safety.

---

## PART 5: LESSONS LEARNED

### 5.1 "General Models aren't enough for Finance" (Bloomberg)
-   GPT-4 is great, but it doesn't know the specific nuance of a "10-K filing footnote".
-   **Lesson**: For high-stakes verticals, **Domain Adaptation** (Pre-training or Fine-tuning) is worth the cost.

### 5.2 "Latency is a Safety Feature" (Roblox)
-   If you detect bullying 5 minutes later, the damage is done.
-   **Lesson**: **Inference Optimization** (CPU serving, quantization) is critical for real-time safety systems.

### 5.3 "Internal Data is the Moat" (Goldman/Morgan Stanley)
-   Everyone has GPT-4. Only Morgan Stanley has 20 years of proprietary research.
-   **Lesson**: The value isn't the model; it's the **RAG Pipeline** connected to unique data.

---

## PART 6: QUANTITATIVE METRICS

| Metric | Result | Company | Context |
| :--- | :--- | :--- | :--- |
| **Model Size** | 50 Billion Params | Bloomberg | BloombergGPT |
| **Adoption** | 98% of Teams | Morgan Stanley | AI Assistant Usage |
| **Daily Volume** | 1.1 Million Hours | Roblox | Voice Chat Processed |
| **Productivity** | 15% Increase | Goldman Sachs | Developer Efficiency |

---

## PART 7: REFERENCES

**Bloomberg (1)**:
1.  BloombergGPT Research Paper (2023-2024)

**Morgan Stanley (1)**:
1.  AI Assistant & RAG Architecture (2024)

**Goldman Sachs (1)**:
1.  Generative AI for Coding (2024)

**Roblox (1)**:
1.  Real-time Voice Safety (2024)

**Unity (1)**:
1.  Muse Chat Architecture (2024)

---

**Analysis Completed**: November 2025  
**Total Companies**: 5 (Bloomberg, Morgan Stanley, Goldman Sachs, Roblox, Unity)  
**Use Cases Covered**: Financial LLMs, Enterprise RAG, Voice Safety, Game Dev  
**Status**: Comprehensive Analysis Complete
