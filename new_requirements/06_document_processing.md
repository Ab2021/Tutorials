# DOCUMENT PROCESSING — PDF, Excel, Scanned Docs, OCR, IDP
## The real-world messy document problem for Italian SME clients

---

## SECTION 1: THE DOCUMENT CHAOS REALITY

### What Italian SME Documents Look Like

Italian SMEs don't have clean, structured data:
- **Fatture (Invoices):** Mix of digital PDFs and scanned PDFs, sometimes handwritten notes
- **Contratti (Contracts):** Long PDFs with complex formatting, tables, footnotes
- **Documenti di trasporto (DDT/Delivery notes):** Often paper-scanned, low quality
- **Email chains:** Long threads with attachments, mixed Italian/English
- **Excel spreadsheets:** Reporting, price lists, inventory — semi-structured
- **Legacy ERPs:** Data exports in CSV, XML, fixed-width formats

The core challenge: **Get clean, structured text from these documents before indexing.**

---

## SECTION 2: PDF PROCESSING — THE FULL STACK

### PDF Types and Appropriate Tools

```
PDF Type 1: Digital/Born-digital PDF
  - Created by software (Word → PDF, Adobe, etc.)
  - Has selectable text layer
  - Easy: extract text directly
  - Tools: PyMuPDF (fastest), pdfplumber (better for tables), pypdf

PDF Type 2: Scanned PDF (image of paper document)
  - Scanned from physical document
  - No text layer — just pixels
  - Requires OCR
  - Tools: Tesseract, Google Document AI, Azure Document Intelligence, GPT-4V

PDF Type 3: Mixed PDF
  - Some pages digital, some scanned
  - Need to detect and handle each type separately
  - Common in Italian SMEs (filing cabinets scanned piecemeal)
```

### Digital PDF Processing

```python
# Option 1: PyMuPDF — Fastest, best for clean text
import fitz  # PyMuPDF

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """Extract text page by page with metadata"""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc):
        # Extract text with layout preservation
        text = page.get_text("text")  # or "blocks" for positional info
        
        # Check if page has text content (distinguish from scanned pages)
        is_scanned = len(text.strip()) < 50  # Too little text = likely scanned
        
        if is_scanned:
            # Extract as image for OCR
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            page_image_bytes = pix.tobytes("png")
            pages.append({
                "page_num": page_num + 1,
                "type": "scanned",
                "image_bytes": page_image_bytes,
                "text": None  # Will be filled by OCR
            })
        else:
            pages.append({
                "page_num": page_num + 1,
                "type": "digital",
                "text": text,
                "image_bytes": None
            })
    
    doc.close()
    return pages

# Option 2: pdfplumber — Better for tables
import pdfplumber

def extract_tables_and_text(pdf_path: str) -> dict:
    """Extract both regular text and tables from PDF"""
    with pdfplumber.open(pdf_path) as pdf:
        result = {"pages": []}
        
        for page in pdf.pages:
            page_data = {
                "page_num": page.page_number,
                "text": page.extract_text(),
                "tables": []
            }
            
            # Extract tables as list of lists
            tables = page.extract_tables()
            for table in tables:
                if table:  # Non-empty table
                    # Convert to dict format with headers
                    headers = table[0]
                    rows = table[1:]
                    table_dict = [dict(zip(headers, row)) for row in rows if row]
                    page_data["tables"].append(table_dict)
            
            result["pages"].append(page_data)
    
    return result
```

### OCR for Scanned Documents

```python
# Option 1: Tesseract (free, local, Italian language support)
import pytesseract
from PIL import Image
import io

def ocr_with_tesseract(image_bytes: bytes, language: str = "ita") -> str:
    """OCR using Tesseract — fully local"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # OCR with Italian language model
    text = pytesseract.image_to_string(
        image,
        lang=language,                    # "ita" for Italian, "ita+eng" for mixed
        config="--oem 3 --psm 1"          # OEM 3 = LSTM, PSM 1 = auto page segmentation
    )
    
    return text

# Setup: install tesseract-ocr + italian language pack
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-ita

# Option 2: GPT-4V / Claude (paid, better quality, handles handwriting)
def ocr_with_gpt4v(image_bytes: bytes) -> str:
    """OCR using GPT-4 Vision — handles handwriting, stamps, poor quality"""
    import base64
    image_b64 = base64.b64encode(image_bytes).decode()
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                },
                {
                    "type": "text",
                    "text": """Extract all text from this document image exactly as it appears.
                    Preserve the structure (tables, lists, headers).
                    This is an Italian business document.
                    Return the extracted text only, no commentary."""
                }
            ]
        }],
        max_tokens=4000
    )
    
    return response.choices[0].message.content

# Option 3: Azure Document Intelligence (managed OCR + structure extraction)
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

def azure_document_intelligence(pdf_path: str) -> dict:
    """Extract structured data using Azure Document Intelligence"""
    client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_DI_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_DI_KEY"))
    )
    
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-invoice",  # Use prebuilt invoice model!
            document=f
        )
    
    result = poller.result()
    
    extracted = {}
    for invoice in result.documents:
        # Azure automatically extracts structured invoice fields
        fields = invoice.fields
        extracted = {
            "vendor_name": fields.get("VendorName", {}).get("value", ""),
            "invoice_id": fields.get("InvoiceId", {}).get("value", ""),
            "invoice_date": str(fields.get("InvoiceDate", {}).get("value", "")),
            "total_amount": fields.get("InvoiceTotal", {}).get("value", {}).get("amount", 0),
            "currency": fields.get("InvoiceTotal", {}).get("value", {}).get("currency_symbol", "EUR"),
            "line_items": [
                {
                    "description": item.get("Description", {}).get("value", ""),
                    "quantity": item.get("Quantity", {}).get("value", 0),
                    "unit_price": item.get("UnitPrice", {}).get("value", {}).get("amount", 0),
                    "amount": item.get("Amount", {}).get("value", {}).get("amount", 0)
                }
                for item in fields.get("Items", {}).get("value", [])
            ]
        }
    
    return extracted
```

### The Intelligent Document Processing (IDP) Pipeline

```python
def full_document_pipeline(file_path: str, client_id: str) -> ProcessedDocument:
    """Complete document processing pipeline"""
    
    # Step 1: Detect file type
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == ".pdf":
        pages = extract_text_pymupdf(file_path)
        
        # Process each page
        full_text_parts = []
        for page in pages:
            if page["type"] == "digital":
                full_text_parts.append(page["text"])
            else:  # Scanned
                if GDPR_LOCAL_ONLY:
                    ocr_text = ocr_with_tesseract(page["image_bytes"])
                else:
                    ocr_text = ocr_with_gpt4v(page["image_bytes"])
                full_text_parts.append(ocr_text)
        
        full_text = "\n\n".join(full_text_parts)
    
    elif file_ext in [".xlsx", ".xls", ".csv"]:
        full_text = extract_excel_text(file_path)
    
    elif file_ext in [".docx", ".doc"]:
        full_text = extract_word_text(file_path)
    
    elif file_ext in [".eml", ".msg"]:
        full_text = extract_email_text(file_path)
    
    else:
        raise UnsupportedFileType(f"Cannot process {file_ext} files")
    
    # Step 2: Clean extracted text
    clean_text = clean_extracted_text(full_text)
    
    # Step 3: Detect document type (invoice, contract, email, etc.)
    doc_type = classify_document(clean_text)
    
    # Step 4: Extract structured fields (if applicable)
    structured_fields = {}
    if doc_type == "invoice":
        structured_fields = extract_invoice_fields(clean_text)
    elif doc_type == "contract":
        structured_fields = extract_contract_metadata(clean_text)
    
    # Step 5: Detect language
    language = detect_language(clean_text)
    
    return ProcessedDocument(
        doc_id=generate_id(file_path, client_id),
        client_id=client_id,
        source_file=file_path,
        doc_type=doc_type,
        language=language,
        raw_text=full_text,
        clean_text=clean_text,
        structured_fields=structured_fields,
        processing_metadata={
            "pages": len(pages) if file_ext == ".pdf" else 1,
            "ocr_used": any(p["type"] == "scanned" for p in pages) if file_ext == ".pdf" else False,
            "processed_at": datetime.utcnow().isoformat()
        }
    )
```

---

## SECTION 3: EXCEL/SPREADSHEET PROCESSING

### The Excel Challenge

Excel files are semi-structured — they have meaning through:
- Cell positions (A1, B1 = headers; A2:B100 = data)
- Formatting (bold = important, color = category)
- Multiple sheets (one per month, one per product line)
- Named ranges and formulas

Simple text extraction loses this structure completely.

```python
import openpyxl
import pandas as pd

def extract_excel_intelligent(excel_path: str) -> str:
    """Convert Excel to LLM-readable text preserving structure"""
    
    wb = openpyxl.load_workbook(excel_path, data_only=True)  # data_only=True reads computed values
    result_parts = []
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        result_parts.append(f"## Sheet: {sheet_name}\n")
        
        # Read with pandas for clean table extraction
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        
        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            continue
        
        # Convert to markdown table for LLM consumption
        markdown_table = df.to_markdown(index=False, floatfmt=".2f", na_rep="")
        result_parts.append(markdown_table)
        result_parts.append("\n")
    
    return "\n".join(result_parts)

# For very large Excel files (price lists, inventory):
def excel_to_chunks(excel_path: str, rows_per_chunk: int = 50) -> list[str]:
    """Break large Excel into chunks for indexing"""
    df = pd.read_excel(excel_path)
    chunks = []
    
    # Add column headers to every chunk for context
    headers = df.columns.tolist()
    
    for start in range(0, len(df), rows_per_chunk):
        chunk_df = df.iloc[start:start + rows_per_chunk]
        chunk_text = f"Columns: {', '.join(headers)}\n\n"
        chunk_text += chunk_df.to_string(index=False)
        chunks.append(chunk_text)
    
    return chunks
```

---

## SECTION 4: EMAIL PROCESSING

### Email Chain Processing

```python
import email
from email.policy import default
import chardet

def extract_email_chain(eml_path: str) -> dict:
    """Extract email with full metadata"""
    with open(eml_path, 'rb') as f:
        raw = f.read()
    
    # Detect encoding (Italian emails often use latin-1 or windows-1252)
    detected_encoding = chardet.detect(raw)['encoding'] or 'utf-8'
    msg = email.message_from_bytes(raw, policy=default)
    
    result = {
        "from": str(msg.get("From", "")),
        "to": str(msg.get("To", "")),
        "cc": str(msg.get("Cc", "")),
        "subject": str(msg.get("Subject", "")),
        "date": str(msg.get("Date", "")),
        "body": "",
        "attachments": []
    }
    
    # Extract body and attachments
    for part in msg.walk():
        content_type = part.get_content_type()
        
        if content_type == "text/plain":
            body = part.get_content()
            if isinstance(body, bytes):
                body = body.decode(detected_encoding, errors='replace')
            result["body"] = body
        
        elif content_type in ["application/pdf", "application/vnd.openxmlformats..."]:
            # Save attachment for separate processing
            filename = part.get_filename()
            content = part.get_content()
            if filename and content:
                result["attachments"].append({
                    "filename": filename,
                    "content_type": content_type,
                    "content": content  # Bytes — process separately
                })
    
    return result

def process_email_for_rag(email_data: dict) -> str:
    """Convert email to LLM-readable format"""
    text = f"""
Email from: {email_data['from']}
To: {email_data['to']}
Date: {email_data['date']}
Subject: {email_data['subject']}

Body:
{email_data['body']}
"""
    return text.strip()
```

---

## SECTION 5: TEXT CLEANING PIPELINE

### The Essential Cleaning Steps

```python
import re
import unicodedata

def clean_extracted_text(raw_text: str, language: str = "it") -> str:
    """Clean extracted text for indexing"""
    
    # 1. Unicode normalization (handles special chars in scanned Italian docs)
    text = unicodedata.normalize("NFKC", raw_text)
    
    # 2. Remove repeated whitespace (very common in PDF extraction)
    text = re.sub(r' {2,}', ' ', text)
    
    # 3. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 4. Remove excess blank lines (keep max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Remove common PDF artifacts (page numbers, headers/footers)
    # Page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Running headers like "COMPANY NAME - CONFIDENTIAL - Page X of Y"
    text = re.sub(r'^.{0,50}pagina?\s+\d+\s+di\s+\d+.{0,50}$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 6. Remove control characters (corruption artifact)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 7. Fix common OCR errors in Italian documents
    ocr_corrections = {
        r'\b0 (?=\d)': 'O ',  # OCR: 0 instead of O
        r'(?<=\d)l(?=\d)': '1',  # OCR: l instead of 1
        'rn': 'm',  # OCR: rn instead of m (context-dependent)
    }
    # Apply carefully — only use well-validated rules
    
    # 8. Normalize currency symbols
    text = text.replace('€', 'EUR ').replace('£', 'GBP ')
    
    # 9. Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def validate_extraction_quality(text: str, filename: str) -> dict:
    """Check if extraction produced usable text"""
    issues = []
    
    if len(text.strip()) < 100:
        issues.append("Too little text extracted (< 100 chars) — likely OCR failure or scanned")
    
    if len(set(text.split())) < 20:
        issues.append("Very low vocabulary diversity — likely repeated pattern or garbage")
    
    if text.count('?') / max(len(text), 1) > 0.1:
        issues.append("High ratio of '?' — likely encoding issue")
    
    word_lengths = [len(w) for w in text.split() if w.isalpha()]
    avg_word_length = sum(word_lengths) / max(len(word_lengths), 1)
    if avg_word_length > 15:
        issues.append("Unusually long words — likely OCR concatenation error")
    
    return {
        "filename": filename,
        "char_count": len(text),
        "word_count": len(text.split()),
        "quality_issues": issues,
        "is_usable": len(issues) == 0
    }
```

---

## SECTION 6: STRUCTURED DATA EXTRACTION WITH LLM

### Invoice Field Extraction

```python
from pydantic import BaseModel, Field
from typing import Optional

class InvoiceLineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    vat_rate: float = Field(description="IVA rate as percentage (e.g., 22.0 for 22%)")
    total: float

class ItalianInvoice(BaseModel):
    numero_fattura: str = Field(description="Invoice number")
    data_fattura: str = Field(description="Invoice date in YYYY-MM-DD format")
    fornitore_nome: str = Field(description="Supplier name")
    fornitore_piva: Optional[str] = Field(description="Supplier VAT number (Partita IVA)")
    fornitore_cf: Optional[str] = Field(description="Supplier fiscal code (Codice Fiscale)")
    cliente_nome: str = Field(description="Customer name")
    cliente_piva: Optional[str] = Field(description="Customer VAT number")
    imponibile: float = Field(description="Net amount (imponibile)")
    iva: float = Field(description="VAT amount (IVA)")
    totale: float = Field(description="Total amount including VAT")
    scadenza: Optional[str] = Field(description="Payment due date in YYYY-MM-DD")
    voci: list[InvoiceLineItem] = Field(description="Line items")

def extract_invoice_with_llm(invoice_text: str) -> ItalianInvoice:
    """Extract structured invoice data using LLM with Pydantic validation"""
    from langchain_core.output_parsers import PydanticOutputParser
    
    parser = PydanticOutputParser(pydantic_object=ItalianInvoice)
    
    prompt = f"""Estrai i dati strutturati da questa fattura italiana.
    
{parser.get_format_instructions()}

Fattura:
{invoice_text}

Se un campo non è presente nel documento, usa null.
Data i importi numerici senza simboli valuta.
"""
    
    response = llm.invoke(prompt)
    
    try:
        invoice_data = parser.parse(response.content)
        return invoice_data
    except Exception as e:
        # Log parsing failure for human review
        logger.error("Invoice extraction failed", error=str(e), text_preview=invoice_text[:200])
        raise ExtractionError(f"Could not extract structured data: {e}")
```

---

## SECTION 7: DOCUMENT CLASSIFICATION

```python
def classify_document(text: str) -> str:
    """Classify document type using LLM"""
    
    # Try keyword classification first (fast, free)
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in ["fattura", "invoice", "imponibile", "iva", "partita iva"]):
        return "fattura"
    elif any(kw in text_lower for kw in ["contratto", "accordo", "clausola", "art.", "articolo"]):
        return "contratto"
    elif any(kw in text_lower for kw in ["documento di trasporto", "ddt", "consegna"]):
        return "ddt"
    elif any(kw in text_lower for kw in ["preventivo", "offerta commerciale", "quotazione"]):
        return "preventivo"
    elif any(kw in text_lower for kw in ["bolla", "ordine di acquisto"]):
        return "ordine"
    
    # Fall back to LLM classification for ambiguous documents
    classification_prompt = f"""Classifica questo documento in una delle seguenti categorie:
fattura, contratto, ddt, preventivo, ordine, email, altro

Risposta: solo la categoria, niente altro.

Documento (prime 500 parole):
{text[:2000]}
"""
    
    doc_type = llm.invoke(classification_prompt).content.strip().lower()
    return doc_type if doc_type in ["fattura", "contratto", "ddt", "preventivo", "ordine", "email"] else "altro"
```

---

## SECTION 8: INTERVIEW ANSWERS ON DOCUMENT PROCESSING

### Q: "An Italian accounting firm has 10,000 PDF invoices. How would you process them?"

> "First question I'd ask: are they digital PDFs or scanned? This determines the processing pipeline.
>
> For digital PDFs: I use PyMuPDF to extract text directly — fast, free, works locally. I also extract tables with pdfplumber for itemized line items. Total processing: maybe 0.1 seconds per invoice.
>
> For scanned PDFs: If they're medium quality, Tesseract OCR handles them locally (Italian language pack). If quality is poor or there's handwriting, I'd use Azure Document Intelligence's prebuilt-invoice model — it handles Italian invoices, extracts structured fields automatically, and handles complex layouts. Cost: about €0.01 per page.
>
> For extraction pipeline overall: document type detection → OCR if needed → text cleaning → structured field extraction (invoice number, date, amounts, supplier) using either Azure Document Intelligence or GPT-4o with Pydantic schemas → validation → store in both relational DB (structured fields for querying) and vector DB (full text for RAG Q&A).
>
> Critical: I validate extraction quality on every document — if less than 100 characters extracted or there are obvious OCR artifacts, I flag for human review rather than silently indexing garbage."

### Q: "What do you do when document extraction quality is poor?"

> "Production document processing always has failure cases. My approach:
>
> First: validate after every processing step. If text is too short, word length distribution is wrong (OCR errors), or character encoding issues exist, flag immediately — don't silently index garbage.
>
> Second: tiered quality thresholds. Documents that pass quality checks go to auto-index. Documents that fail go to a review queue with the specific issue noted.
>
> Third: fallback strategies. If Tesseract OCR fails, try GPT-4V (better at poor quality scans). If both fail, flag the document with instructions for the client to re-scan at higher resolution.
>
> Fourth: human review sampling. Even for 'passing' documents, spot-check 5% against original PDFs to catch systematic errors the quality checks didn't catch.
>
> Never deploy a document processing pipeline without showing the client examples of its output on their actual documents — surprises in production are much more damaging than early discovery of issues."
