# LLMs for Underwriting (Part 1) - Submission Triaging & Risk Assessment - Theoretical Deep Dive

## Overview
"Underwriters spend 40% of their time just finding the data to make a decision."
The modern Underwriter is buried in PDFs: Loss Runs, Financial Statements, Engineering Reports.
**LLMs** act as the "Bionic Arm" for Underwriters, automating the **Ingestion** and **Triaging** of submissions so humans can focus on the **Risk Selection**.

---

## 1. Conceptual Foundation

### 1.1 The Submission Tsunami

*   **Problem:** A Commercial Carrier receives 1,000 submissions/day via email.
*   **Reality:** Only 100 fit the "Appetite". Only 10 will be won.
*   **Old Way:** Manual review. (Slow, High Expense Ratio).
*   **New Way (GenAI):**
    1.  **Ingest:** Read email + attachments.
    2.  **Extract:** Revenue, SIC Code, Claims History.
    3.  **Score:** "Green" (Quote Auto), "Yellow" (Refer), "Red" (Decline).

### 1.2 The "Underwriting Assistant" Paradigm

*   **Not Replacement:** The goal isn't to replace the Underwriter.
*   **Augmentation:** The LLM prepares the "Deal Memo".
    *   *Summary:* "This is a \$50M Revenue Construction firm in Texas."
    *   *Flag:* "Note: They had a large fire loss in 2019."
    *   *Recommendation:* "Appetite: High. Pricing Guidance: +5%."

---

## 2. Mathematical Framework

### 2.1 Zero-Shot Classification for Appetite

*   **Task:** Map a vague description to a specific NAICS/SIC Code.
*   **Input:** "We do roof repair and occasional plumbing."
*   **Prompt:** "Classify this business into one of the following NAICS codes: [238160 (Roofing), 238220 (Plumbing), ...]"
*   **Output:** `{"Primary": "238160", "Secondary": "238220"}`.
*   **Math:** $\text{argmax}_C P(C | \text{Description})$.

### 2.2 RAG (Retrieval Augmented Generation) for Guidelines

*   **Problem:** Underwriting Guidelines change weekly.
*   **Solution:** Store guidelines in a Vector Database.
*   **Query:** "Can we write a wood-frame hotel in Florida?"
*   **Retrieval:** Finds "Guideline 4.2: Coastal Wood Frame -> Prohibited."
*   **Generation:** "No, Guideline 4.2 prohibits this due to wind risk."

---

## 3. Theoretical Properties

### 3.1 Hallucination Control in Data Extraction

*   **Risk:** The LLM "invents" a revenue figure if it can't find one.
*   **Mitigation:** **Grounding**.
    *   Require the model to return the *page number* and *bounding box* of the extracted data.
    *   If it can't point to the source, return `null`.

### 3.2 Multi-Agent Systems

*   **Concept:** Break the underwriting task into specialized agents.
    *   **Agent A (Financials):** Reads Balance Sheet. Calculates Current Ratio.
    *   **Agent B (Loss Control):** Reads Safety Report. Checks for sprinklers.
    *   **Agent C (Synthesizer):** Combines A + B into a final Risk Score.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The "Submission Ingestor" (Python)

```python
import fitz # PyMuPDF
from langchain.chat_models import ChatOpenAI

# 1. Extract Text from PDF
doc = fitz.open("submission.pdf")
text = ""
for page in doc:
    text += page.get_text()

# 2. Define Schema (Structured Output)
schema = {
    "properties": {
        "InsuredName": {"type": "string"},
        "Revenue": {"type": "integer"},
        "LossHistory": {
            "type": "array",
            "items": {"type": "object", "properties": {"Year": {"type": "integer"}, "Amount": {"type": "integer"}}}
        }
    },
    "required": ["InsuredName", "Revenue"]
}

# 3. LLM Extraction
llm = ChatOpenAI(model="gpt-4-0613").bind(functions=[{"name": "extract", "parameters": schema}])
response = llm.invoke(f"Extract data from: {text[:4000]}") # Truncate for token limit

print(response.additional_kwargs["function_call"]["arguments"])
```

### 4.2 Appetite Check Logic

```python
def check_appetite(data):
    # Hard Rules (Deterministic)
    if data['Revenue'] > 100_000_000:
        return "Refer to Large Accounts"
    
    # Fuzzy Rules (LLM)
    if "Coal" in data['Description'] or "Mining" in data['Description']:
        return "Decline (ESG)"
    
    return "Quote"
```

---

## 5. Evaluation & Validation

### 5.1 The "Underwriter Turing Test"

*   **Test:** Give an Underwriter two Deal Memos.
    *   Memo A: Written by a Junior Underwriter.
    *   Memo B: Written by the LLM.
*   **Goal:** The Senior Underwriter cannot tell the difference (or prefers the LLM).

### 5.2 Precision/Recall of Extraction

*   **Metric:** Compare LLM extracted values to "Ground Truth" (manually keyed data).
*   **Target:** > 99% Accuracy for "Critical Fields" (Revenue, Address).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "PDF Table" Nightmare**
    *   *Problem:* LLMs are bad at reading complex, multi-column tables in PDFs (e.g., Loss Runs).
    *   *Fix:* Use OCR tools specialized for tables (Amazon Textract, Azure Form Recognizer) *before* sending text to the LLM.

2.  **Trap: Bias in Unstructured Text**
    *   *Scenario:* The Engineering Report mentions "Neighborhood decline".
    *   *Risk:* The LLM interprets this as a Redlining signal.
    *   *Fix:* Explicitly instruct the model to ignore demographic/geographic descriptors unless relevant to physical peril.

---

## 7. Advanced Topics & Extensions

### 7.1 "Chat with Your Data"

*   **Feature:** Underwriter opens a submission and asks:
    *   "Does this insured have a cyber policy with us?"
    *   "What was the cause of the 2021 loss?"
*   **Tech:** RAG over the specific submission folder.

### 7.2 Portfolio-Level Insights

*   **Idea:** Use LLMs to cluster the *entire book* of submissions.
*   **Insight:** "We are seeing a 20% spike in 'Solar Panel Installer' submissions in Arizona."
*   **Action:** Launch a specific product for that niche.

---

## 8. Regulatory & Governance Considerations

### 8.1 Adverse Action Notices

*   **Regulation:** If you decline a submission, you must provide a reason.
*   **LLM Role:** The LLM must generate the decline letter citing the specific underwriting rule (e.g., "Declined due to lack of 3 years loss history").

---

## 9. Practical Example

### 9.1 Worked Example: The "Smart Inbox"

**Scenario:**
*   **Inflow:** Broker sends an email with 5 attachments (App, Loss Runs, Financials).
*   **Process:**
    1.  **Classifier:** "This is a New Business Submission."
    2.  **Extractor:** "Insured: ABC Corp. Rev: \$10M. Losses: \$0."
    3.  **Appetite:** "Green."
    4.  **Enrichment:** API call to Dun & Bradstreet verifies credit score.
    5.  **Draft Quote:** System pre-fills the Rating Engine.
*   **Output:** Underwriter opens the system and sees a "Ready to Bind" quote. They just review and hit "Send".
*   **Efficiency:** 90% reduction in manual data entry.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Ingestion** is the first battleground.
2.  **Structured Output** (JSON) is essential for downstream systems.
3.  **RAG** keeps the LLM aligned with Guidelines.

### 10.2 When to Use This Knowledge
*   **COO:** "How do we scale GWP without hiring more Underwriters?"
*   **Chief Underwriting Officer:** "How do I ensure consistent decision making across 500 underwriters?"

### 10.3 Critical Success Factors
1.  **Human Review:** The Underwriter is the "Pilot", the AI is the "Co-Pilot".
2.  **Data Quality:** Garbage in (bad OCR), Garbage out (bad extraction).

### 10.4 Further Reading
*   **BCG:** "The Generative AI Opportunity in Insurance Underwriting".

---

## Appendix

### A. Glossary
*   **OCR:** Optical Character Recognition.
*   **SIC/NAICS:** Standard Industry Classification codes.
*   **Loss Run:** Report of past claims.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Classification** | $\text{argmax} P(C|D)$ | Triage |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
