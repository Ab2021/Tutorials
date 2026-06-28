# DOCUMENT PROCESSING & INGESTION: THE MASTERCLASS (v1)
## Architecting the ingestion of chaotic SME data (No Code)

> **Critical Context:** Junior AI Engineers assume data arrives as clean JSON. Lead AI Engineers know that for Italian SMEs, data arrives as 15-year-old scanned PDFs, massive Excel grids, and deeply nested email chains. If your ingestion pipeline fails to clean and structure this data, your expensive LLM will confidently generate hallucinations. This document covers the architecture of multi-modal ingestion.

---

## SECTION 1: THE PDF PARSING PIPELINE

A PDF is not a text file. It is a set of visual instructions for drawing characters on a screen. Extracting semantic meaning from it requires an intelligent routing architecture.

### The Document Routing Architecture
When a PDF enters the system, it must be triaged. 
1.  **The Digital Fast-Path:** A lightweight script (using a library like PyMuPDF) scans the document to check if it has a selectable text layer. If yes, it extracts the text directly. This takes milliseconds and costs €0.
2.  **The OCR Slow-Path:** If the document has no text layer (it is an image of a scanned paper), the system routes it to an Optical Character Recognition (OCR) engine.
    -   *Local Tesseract:* Used for high-volume, low-budget, or highly confidential documents. It requires installing Italian language packs to handle accents (`à, è, ì, ò, ù`).
    -   *Cloud Vision API:* (e.g., Azure Document Intelligence, GPT-4o-Vision). Used for highly degraded scans, handwriting, or complex multi-column layouts. It is slow and costs money, so it is strictly used as a fallback.

### The "Dirty OCR" Problem & Normalization
Scanned Italian documents frequently produce OCR artifacts (e.g., confusing "0" with "O", or breaking a single word across two lines with a hyphen). If you index this directly, vector search will fail because the embedding model will not recognize the broken words.
-   **Architectural Fix:** Implement a Post-OCR Normalization Layer. Before the text hits the chunking algorithm, run a sequence of deterministic scripts. Remove excessive whitespace, rejoin hyphenated words, normalize unicode characters to a standard format (NFC), and run a regex pass to standardize currency symbols (e.g., converting `€` and `Euro` to a standard `EUR`).

---

## SECTION 2: THE TABLE EXTRACTION CRISIS

LLMs are fundamentally language models. They read left-to-right. If you use standard text extraction on a PDF table, it will read across the columns, jumbling the product name from Column A with the price from Column B and the quantity from Column C.

### Architecting for Tables
1.  **Detection:** Use a layout-aware parser (like pdfplumber or Azure Document Intelligence) that specifically identifies the geometric bounding boxes of tables.
2.  **Serialization:** Once the table grid is detected, you must serialize it into a format the LLM can reason about. 
    -   *Markdown Tables:* Convert the grid into a Markdown table structure. LLMs have been heavily trained on Markdown and can easily trace relationships vertically and horizontally within this format.
    -   *Row-by-Row Serialization:* For very wide tables, convert each row into a self-contained sentence: `[Product: Widget A | Price: 10 EUR | Quantity: 5]`.
3.  **Chunking Implications:** You must configure your chunking algorithm to *never* split a table in half. If a table is massive and must be split, the architecture must forcibly inject the column headers into every newly created chunk, otherwise, the LLM will see a list of numbers and have no idea what they represent.

---

## SECTION 3: EXCEL AND SPREADSHEET INGESTION

SMEs run on Excel. However, feeding a 10,000-row Excel file into an LLM context window is mathematically impossible and architecturally flawed.

### The "Data-to-Text" vs "Code-Generation" Architecture
You must architect based on the user's intent.

**Scenario A: The user wants to search for specific rows.**
-   **Solution:** Deconstruct the Excel file. Drop all empty rows and columns. Convert each remaining row into a JSON object or a serialized string. Embed each row independently into the Vector Database. When the user asks "What is the price of Widget A?", semantic search finds the exact row.

**Scenario B: The user wants to perform mathematical aggregations ("What is the total revenue for Q3?").**
-   **Solution:** Do NOT use RAG. LLMs are terrible at math. Architect a **Data Analyst Agent**.
-   The Excel file is loaded into a secure, backend Pandas DataFrame.
-   When the user asks the question, the LLM does not read the Excel file. Instead, the LLM is instructed to write a Python Pandas script that will calculate Q3 revenue.
-   The Orchestrator executes the generated Python script in a secure sandbox against the DataFrame, and returns the mathematically perfect answer to the user.

---

## SECTION 4: INTELLIGENT DOCUMENT PROCESSING (IDP)

For use cases like processing Invoices or Legal Contracts, the goal is not to "search" the document, but to extract structured data (JSON) from messy unstructured text.

### The Structured Output Architecture
Historically, companies used brittle regex patterns to extract data from invoices. If a vendor moved the "Total" from the top right to the bottom left, the regex broke.

**The Modern AI Approach:**
1.  **Raw Extraction:** Extract all text from the invoice using the pipeline in Section 1.
2.  **Schema Definition:** Define a strict Pydantic JSON schema representing the data you need (e.g., `invoice_number`, `vendor_name`, `total_amount_eur`).
3.  **The Extraction Call:** Send the raw text and the JSON schema to an LLM (like GPT-4o) using the "Structured Outputs" API. Because the LLM understands semantics, it can find the total amount regardless of where it is physically located on the page. The API mathematically guarantees the output will perfectly match your JSON schema.
4.  **The Validation Layer (Crucial):** Never trust the LLM implicitly. Implement a hardcoded validation layer immediately after the LLM call. If the LLM extracts `Subtotal: 100`, `Tax: 20`, `Total: 150`, the deterministic Python code runs the math. If `100 + 20 != 150`, the extraction is flagged as a hallucination and routed to a human for manual review.

---

## SECTION 5: MASSIVE INTERVIEW Q&A BANK (DOCUMENT PROCESSING)

### Q1: An accounting firm has 5,000 scanned PDF invoices. Running them through GPT-4o-Vision for data extraction costs too much. How do you redesign this to be cost-effective?
**Strategy:** Implement a multi-tier routing pipeline.
**Answer:** "I would never route all 5,000 scans to a premium Vision API. I would architect a Triage Pipeline. 
First, I run a fast, free local script to check for digital text layers. The 20% that are digital bypass OCR entirely. 
For the remaining 80%, I route them to a local, free Tesseract OCR instance. I run a heuristic quality check on the Tesseract output (e.g., checking the ratio of alphanumeric characters to special symbols). 
If the local OCR output is clean, I send that text to a cheap model (GPT-4o-mini) for structured extraction. If the local OCR output is garbage (indicating bad handwriting or complex layouts), ONLY THEN do I route that small, problematic subset to the expensive premium Vision API. This tiered architecture delivers 99% accuracy while slashing API costs by over 80%."

### Q2: You are building a system to process email threads for a customer support team. The vector database is retrieving terrible results. Why?
**Strategy:** Explain the "Nested Email Problem" and thread flattening.
**Answer:** "The issue is that email threads repeat the entire conversation history at the bottom of every new reply. If an email thread has 10 replies, and you ingest the raw `.eml` files, the original customer complaint is indexed 10 separate times in the Vector DB. This destroys retrieval precision because the database is flooded with duplicate, noisy vectors. 
To fix this architecturally, I would implement an Email Flattening pipeline before ingestion. I would use a parsing library to strictly identify and strip out all quoted replies, signatures, and legal disclaimers. We only embed the net-new text of each email. Additionally, I would separate attachments, process them individually, and map them back to the parent email using metadata UUIDs."

### Q3: A client wants to extract the 'Limitation of Liability' clause from 100 different legal contracts. The contracts are all formatted completely differently. How do you build this?
**Strategy:** Shift from Keyword Search to LLM Structured Extraction.
**Answer:** "Legacy regex or keyword search will fail here because the clause might be called 'Liability', 'Indemnification', or simply be an unnamed paragraph. 
I would architect a Structured IDP (Intelligent Document Processing) pipeline. I would parse each contract into large chunks. I would define a Pydantic schema with a single field: `limitation_of_liability_text`. I would iterate through the chunks, passing each one to an LLM with the instruction: 'If this chunk contains a limitation of liability clause, extract the exact text. If not, return null.' 
Because the LLM understands semantics, it will identify the clause based on its meaning, completely ignoring the structural formatting of the document. The results are collected, verified by a validation script, and saved to a database."

### Q4: How do you handle non-English languages, specifically Italian business documents, in an embedding pipeline?
**Strategy:** Model selection and Unicode normalization.
**Answer:** "Using default English-centric models on Italian documents is a massive architectural error. 
First, in the normalization pipeline, I must ensure full support for UTF-8 encoding to prevent accents (`è`, `à`) from being corrupted into garbage characters (`Ã¨`), which destroys the semantic meaning before it even hits the model. 
Second, I must select a natively multilingual embedding model. While OpenAI's models are good, if the client requires on-premise execution, I would explicitly deploy a model like `multilingual-e5-large` or a specific Italian model from the huggingface `sentence-transformers` library. These models are specifically trained on cross-lingual data and will cluster Italian legal terms correctly in the vector space."
