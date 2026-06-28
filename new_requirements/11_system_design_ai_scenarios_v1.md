# AI SYSTEM DESIGN SCENARIOS FOR ITALIAN SMEs (v1 - CONCEPTUAL & ARCHITECTURAL)
## Real-world whiteboard scenarios, trade-offs, and architectures (No Code)

---

## 1. HOW TO APPROACH SYSTEM DESIGN INTERVIEWS

When an interviewer says, "Design an AI system for X," do not jump immediately to vectors and LangGraph. Use the **7-Block Framework** tailored for Consulting:

1.  **Business Goal & Baseline:** What is the manual process? What is the success metric?
2.  **Constraints:** Data privacy (GDPR), budget, latency, legacy software integration.
3.  **Data Ingestion & Cleaning:** How does messy real-world data enter the system?
4.  **The AI Engine (Core):** RAG, Classification, Extraction, or Agents.
5.  **Execution & Delivery (UI/UX):** How does the human interact with it? (Copilot vs. Autonomous).
6.  **Evaluation & Guardrails:** How do we prove it works and prevent disasters?
7.  **Infrastructure & Cost:** Cloud vs. On-Premise, caching, scaling.

---

## SCENARIO A: THE LEGAL CONTRACT ANALYZER

**Prompt:** An Italian commercial law firm wants to build an internal tool. Lawyers currently spend 10 hours a week reading 100-page lease agreements to check if they comply with a specific set of new property laws. 

### 1. Business Goal & Constraints
-   **Goal:** Reduce manual reading time; highlight non-compliant clauses.
-   **Constraints (Critical):** Extreme data privacy. Client contracts cannot be sent to public APIs. Absolute accuracy required (hallucinations cause lawsuits).
-   **Baseline:** Paralegals manually reading and highlighting PDFs.

### 2. Architecture & Data Flow
-   **Ingestion Layer:** Paralegal uploads a PDF into a secure local web interface.
-   **Processing (OCR):** Because legal PDFs might be scanned, we run a local OCR process (e.g., Tesseract with Italian models) to guarantee full text extraction.
-   **AI Core (Local LLM):** Since data cannot leave the firm, we deploy a local inference server running a quantized model (e.g., Llama-3-70B if they have a GPU, or 8B if CPU bound). 
-   **Prompting Strategy (Map-Reduce):** Instead of RAG, we use a chunked analysis pipeline. The system breaks the 100-page document into sections (Articles/Clauses). It passes each clause to the local LLM with the prompt: *"Analyze this clause against Property Law X. Does it comply? Extract the conflicting text."*
-   **UI Delivery:** A side-by-side view. The original PDF on the left. The AI's flagged clauses on the right. 

### 3. Evaluation & Guardrails
-   **The Guardrail:** The system is purely an "Augmented Reader." It does not rewrite the contract. It only flags potential issues for a lawyer to verify. 
-   **Evaluation:** We establish a "Golden Dataset" of 50 historically problematic contracts. Before deploying a new version of the prompt or model, the system must achieve 99% Recall (it must catch every violation the humans caught).

---

## SCENARIO B: THE LOGISTICS EMAIL AUTOMATION

**Prompt:** A mid-sized logistics company receives 500 emails a day from clients asking "Where is my shipment?" or "Can I change the delivery address?" A human dispatcher reads these, looks up the ID in their legacy ERP, and replies. They want to automate this.

### 1. Business Goal & Constraints
-   **Goal:** Reduce response time from hours to minutes; free up the dispatcher.
-   **Constraints:** The ERP is a 15-year-old on-premise SQL database with no API. Hallucinating a delivery date will infuriate customers.
-   **Baseline:** Dispatcher copy-pasting tracking numbers.

### 2. Architecture & Data Flow
-   **Ingestion:** An email listener (via IMAP or a webhook like n8n) triggers when a new email arrives in `support@logistics.it`.
-   **AI Core 1 (Classification & Extraction):** A fast, cheap cloud model (GPT-4o-mini) reads the email. It classifies the intent (Status Request vs. Address Change) and extracts entities (Tracking Number: IT123456).
-   **Integration Layer (The ERP Hack):** Since there is no API, we write a Python microservice that acts as an adapter. It takes the tracking number, runs a direct SQL `SELECT` query against the legacy ERP database (Read-Only access), and formats the result as JSON.
-   **AI Core 2 (Generation):** The JSON data (Status: In Transit, Location: Milan) is passed back to the LLM to draft a polite, professional reply in Italian.
-   **Delivery (Human-in-the-Loop):** The email is NOT sent automatically. It is saved in the dispatcher's "Drafts" folder. The dispatcher reviews and hits "Send."

### 3. Evaluation & Guardrails
-   **Guardrail:** Read-only access to the ERP. The AI cannot modify database records (e.g., it cannot actually change the address; it drafts a reply saying "A human will contact you to confirm the address change").
-   **Evaluation:** Track the "Send-As-Is" rate. If the dispatcher edits the AI's drafted email less than 10% of the time, the system is highly successful.

---

## SCENARIO C: THE MANUFACTURING EQUIPMENT TROUBLESHOOTER

**Prompt:** A factory makes industrial packaging machines. When a machine breaks down at a client site, technicians spend hours searching through PDF manuals, wiring diagrams, and past maintenance logs to figure out the fix. Design an AI assistant for the technicians.

### 1. Business Goal & Constraints
-   **Goal:** Reduce Mean Time To Repair (MTTR) by surfacing technical answers instantly.
-   **Constraints:** Technicians are on the factory floor using mobile devices (often with bad cell reception). Manuals contain highly technical jargon and diagrams.
-   **Baseline:** Ctrl+F through massive PDFs.

### 2. Architecture & Data Flow
-   **Ingestion (The Knowledge Base):** We process three data sources: PDF Manuals, Past Maintenance Logs (Jira/Excel), and CAD diagrams (metadata only). 
-   **Chunking Strategy (Critical):** Standard token chunking will destroy technical manuals. We implement Semantic Document Chunking—chunking specifically by headers, error codes, and chapter boundaries. 
-   **Vector Database:** Qdrant or Weaviate, hosted in a stable cloud environment. 
-   **Retrieval Strategy (Hybrid RAG):** Pure semantic search fails on specific part numbers (e.g., "Error Code E-404" vs "Error Code E-405"). We must use Hybrid Search. We combine Dense Vector search (for conceptual queries like "machine making grinding noise") with BM25 Keyword Search (for exact part numbers). 
-   **Delivery:** A mobile-friendly progressive web app (PWA) where a technician can type or voice-dictate their issue.

### 3. Evaluation & Guardrails
-   **Guardrail:** The AI must explicitly cite its sources with page numbers. "Replace the valve (Source: Maintenance Manual, Page 42)." If the LLM generates a fix without a source, the system is instructed to append a warning: "No documentation found for this procedure. Consult senior engineering."
-   **Evaluation:** Context Precision is the key metric here. When a technician searches an error code, the correct page of the manual MUST be in the top 3 retrieved chunks, otherwise the system is useless.

---

## SCENARIO D: THE ACCOUNTING INVOICE RECONCILIATION

**Prompt:** An accounting firm processes thousands of invoices a month for their clients. They want an AI to extract data from vendor invoices (which come in 100 different layouts) and format it for their accounting software.

### 1. Business Goal & Constraints
-   **Goal:** Eliminate manual data entry.
-   **Constraints:** High volume. High accuracy requirement (a missed decimal point costs money). PII and Financial data present. 
-   **Baseline:** Manual typing into an Excel sheet.

### 2. Architecture & Data Flow
-   **Ingestion:** Batch processing pipeline runs overnight on a secure cloud storage bucket.
-   **AI Core (Structured Output):** This is not a conversational AI; this is an Information Extraction pipeline. We use an LLM configured with "Structured Outputs" (e.g., forcing the LLM to return data matching a strict Pydantic JSON schema). 
-   **Schema Design:** The LLM is forced to extract: `invoice_number`, `date`, `vendor_name`, `vat_total`, `grand_total`. 
-   **Validation Layer:** Code-based validation runs *after* the LLM. It checks the math: does `subtotal + vat == grand_total`? If the math fails, the LLM hallucinated, or the OCR failed.
-   **Delivery:** Data is pushed to a staging table in the accounting software. Invoices that failed the math check are flagged in red for human review. 

### 3. Cost & Infrastructure
-   **Cost Optimization:** Processing 10,000 invoices via standard API calls is expensive. We architect this to use the OpenAI Batch API. We bundle the invoices, send them at night, and get the results the next morning for 50% of the cost.
-   **Evaluation:** Precision is far more important than Recall. We would rather the AI say "I cannot read this invoice" (low recall) than extract the wrong total amount (low precision). We evaluate against a test set of 200 highly complex, messy invoices.
