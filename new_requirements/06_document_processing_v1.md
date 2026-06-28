# DOCUMENT PROCESSING & INGESTION ARCHITECTURE (v1 - CONCEPTUAL)
## How to handle messy, real-world SME documents before AI touches them (No Code)

---

## 1. THE SME DATA REALITY: WHY AI FAILS BEFORE IT STARTS

If you feed garbage into a state-of-the-art LLM, you will get highly articulate, confidently hallucinated garbage out. 

The biggest failure point in AI consulting for SMEs is not the LLM choice; it is the document ingestion pipeline. SMEs do not have clean APIs or pristine JSON data. They have 20-year-old scanned PDFs, complex Excel spreadsheets with merged cells, and emails with massive embedded email trails. Your architecture must normalize this chaos.

---

## 2. THE MULTI-MODAL PDF PROCESSING ARCHITECTURE

A PDF is not a text file; it is a visual layout instruction file. A PDF can contain pure digital text, purely scanned images of text, or a mix of both. 

### The Routing Pipeline
When a PDF enters the system, the architecture must route it dynamically based on its structural composition.
1.  **The Digital Fast-Path:** If the PDF is digitally born (e.g., exported from Word), the system uses a fast, lightweight library to extract the text layer directly. This is fast, cheap, and highly accurate.
2.  **The OCR Slow-Path:** If the system detects a scanned image (e.g., no selectable text layer), it routes the document to an Optical Character Recognition (OCR) engine. For SMEs, this is often Tesseract (run locally to save costs and ensure privacy) or a cloud service like Azure Document Intelligence for complex layouts.
3.  **The Table Extraction Engine:** Tables destroy standard text extractors. Reading a table left-to-right across columns jumbles the data. The architecture must detect tables, extract them structurally (maintaining row/column relationships), and convert them into a format the LLM can reason about (like Markdown or JSON).

### Handling the "Dirty OCR" Problem
Scanned Italian documents often misinterpret characters (e.g., confusing "0" with "O", or merging characters). 
-   **Architectural Fix:** Implement a Post-OCR Cleaning Layer. Before the text hits the vector database, run a sequence of normalization scripts. This removes massive whitespace blocks, normalizes unicode characters, and flags documents that fall below a basic readability threshold (e.g., too many non-alphanumeric characters) for manual human review, rather than silently indexing garbage.

---

## 3. MASTERING THE SPREADSHEET (EXCEL/CSV)

LLMs are fundamentally language models; they struggle profoundly with large, grid-based mathematical data. You cannot simply extract all text from a 10,000-row Excel file, dump it into a RAG prompt, and expect the LLM to calculate "Total Revenue for Q3."

### The "Data-to-Text" Architectural Pattern
1.  **Deconstruction:** The system breaks the spreadsheet down by sheets, dropping empty rows and columns.
2.  **Serialization:** For small tables, convert the grid into Markdown tables. LLMs understand Markdown natively and can trace relationships vertically and horizontally.
3.  **The Row-Based Strategy:** For larger tables (e.g., a massive product catalog), convert each row into a self-contained sentence or JSON object containing the column headers. For example, instead of a raw grid, the LLM receives: `[Product: "Widget A", Price: "€10", Stock: "45"]`.
4.  **The Agentic Data Analysis Route:** If the client wants to perform complex math on the Excel file, do not use RAG. Architect a "Data Analyst Agent." Give the LLM a tool capable of executing Python code. The LLM writes a Pandas script to query the Excel file dynamically, runs the script in a secure sandbox, and returns the mathematically accurate answer.

---

## 4. TAMING THE EMAIL CHAIN

SME communication lives in endless email threads (often saved as `.eml` or `.msg` files). 

### The Information Extraction Pipeline
1.  **Metadata Separation:** The system must strictly parse the metadata (Sender, Date, CC) away from the Body. If this leaks into the main text, the LLM will get confused by dates and names.
2.  **Thread Flattening:** An email thread repeats the entire conversation history at the bottom of every reply. If you index the raw file, you will index the same conversation 15 times, destroying your vector database's precision. The architecture must strip out previous replies, indexing only the net-new text from each message in the chain.
3.  **Attachment Routing:** Emails contain attachments. The architecture must recursively strip attachments, identify their file types (PDF, Excel, Word), and route them back into the top of the multi-modal ingestion pipeline, maintaining a parent-child relationship link in the database.

---

## 5. INTELLIGENT DOCUMENT PROCESSING (IDP) 

For tasks like Invoice Processing or Contract Extraction, the goal is not to "search" the document, but to extract structured data (JSON) from unstructured text.

### The Structured Output Architecture
1.  **Schema Definition:** You define a strict schema (e.g., `InvoiceNumber`, `VendorName`, `TotalAmount`).
2.  **The Extraction Call:** The LLM is prompted to read the cleaned text and map the findings strictly to the schema. Modern APIs enforce "Structured Outputs," ensuring the LLM physically cannot return data outside the requested JSON format.
3.  **The Code-Based Validator:** Never trust the LLM. Implement a hardcoded validation layer immediately after the LLM. If the LLM extracts `Subtotal: 100`, `Tax: 20`, `Total: 150`, the validation layer runs the math. If `100 + 20 != 150`, the extraction is flagged as a hallucination and routed to a human for review.

---

## 6. INTERVIEW Q&A DRILL-DOWN: DOCUMENT PROCESSING

**Q: A client gives you 50,000 historical PDFs containing a mix of legal contracts and scanned handwritten notes. How do you design an ingestion pipeline that doesn't cost a fortune in OCR fees?**
**Strategy:** Implement a routing and classification architecture.
**Answer:** "Sending 50,000 PDFs to a premium OCR service like Azure Document Intelligence would be incredibly expensive. I would architect a Triage Pipeline. First, a fast, local script parses every PDF to check for a digital text layer. The 80% that are digital bypass OCR entirely and are processed locally for free. The remaining 20% are scanned. For those, I run a local open-source OCR (Tesseract). I run a quick heuristic on the output—if the text is mostly garbage (indicating complex handwriting), only that tiny fraction is routed to the expensive, high-quality cloud OCR API. This tiered architecture saves the client 95% of the processing cost while maintaining high quality."

**Q: Your RAG system is failing because the LLM cannot accurately answer questions about a massive pricing table inside a PDF. How do you fix this?**
**Strategy:** Explain Table Extraction and Semantic Chunking.
**Answer:** "Standard text extraction ruins tables because it reads straight across columns, jumbling prices with product names. First, I would implement a layout-aware PDF parser (like pdfplumber or Azure Document Intelligence) to detect the table structure. Once detected, I do not just dump the table into text. I convert it into Markdown format, which LLMs comprehend exceptionally well. Furthermore, I ensure the chunking strategy respects table boundaries—a table should never be split down the middle across two vector chunks. If the table is massive, I inject the column headers into every single chunk so the LLM never loses the context of what a specific number means."

**Q: How do you handle extracting data from an Italian invoice where the layout changes depending on the vendor?**
**Strategy:** Emphasize LLMs over legacy template matching.
**Answer:** "Historically, OCR systems required us to draw bounding boxes for every vendor's specific invoice template. This is unmaintainable for an SME with 500 vendors. I use an LLM-based IDP (Intelligent Document Processing) approach. I extract the raw, messy text from the invoice, regardless of layout. I then pass that text to an LLM alongside a strict JSON schema representing the fields we need (e.g., P.IVA, Imponibile, Totale). Because LLMs understand semantics, they can find the 'Total Amount' whether it is at the top right, bottom left, or called 'Totale Fattura' vs 'Importo Dovuto'. I then wrap this in a code-based math validator to catch hallucinations."
