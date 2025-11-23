# Day 96: Legal & Compliance Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why is "Recall" more important than "Precision" in E-Discovery?

**Answer:**
*   **Recall:** Finding *all* relevant documents.
*   **Precision:** Finding *only* relevant documents.
*   In law, missing a relevant document (Low Recall) can mean losing the case or sanctions. Showing a few extra irrelevant documents (Low Precision) is annoying but acceptable.

#### Q2: How do you handle "Context Window" limits with 100-page contracts?

**Answer:**
*   **Map-Reduce:** Summarize each section, then combine.
*   **RAG:** Retrieve only the clauses relevant to the question.
*   **Long-Context Models:** Use Gemini 1.5 Pro (1M context) or Claude 3 (200k) to fit the whole contract.

#### Q3: What is "Hallucination" in a legal context?

**Answer:**
Citing a case that doesn't exist (e.g., *Mata v. Avianca*).
*   **Cause:** The model predicts the *pattern* of a citation but invents the names.
*   **Fix:** Tool use. The model must query a Case Law Database (Westlaw/Lexis) to verify the citation exists before outputting it.

#### Q4: How do you ensure data privacy?

**Answer:**
*   **SOC2 / HIPAA:** Compliance certifications.
*   **Zero-Training Agreement:** Contractual guarantee from OpenAI/Azure that data won't be used to train models.
*   **VPC Peering:** Private connection to the API.

### Production Challenges

#### Challenge 1: OCR Errors

**Scenario:** Contract is a scanned PDF. OCR reads "Rent is $1000" as "Rent is $1000".
**Root Cause:** Bad source quality.
**Solution:**
*   **Human Verification:** Flag low-confidence OCR regions for human review.
*   **Multimodal Models:** Use GPT-4V to read the image directly, bypassing traditional OCR.

#### Challenge 2: Conflicting Clauses

**Scenario:** Section 1 says "No termination". Section 10 says "Termination allowed".
**Root Cause:** Badly written contract.
**Solution:**
*   **Conflict Detection:** Ask the agent specifically: "Are there any contradictions between the sections?"

#### Challenge 3: Changing Laws

**Scenario:** The agent advises based on 2021 laws. The law changed in 2024.
**Root Cause:** Training data cutoff.
**Solution:**
*   **RAG:** Always retrieve the *current* statute from a live database. Never rely on parametric memory for laws.

### System Design Scenario: Automated NDA Reviewer

**Requirement:** Review NDAs uploaded by sales team. Approve standard ones, flag risky ones.
**Design:**
1.  **Playbook:** Define "Standard Terms" (e.g., Jurisdiction = NY/DE, Term < 5 years).
2.  **Extraction:** Extract Jurisdiction, Term, Confidentiality Scope.
3.  **Logic:**
    *   If matches Playbook -> Auto-Approve.
    *   If mismatch -> Flag for Legal Team with a note ("Jurisdiction is California").
4.  **Audit Trail:** Log every decision.

### Summary Checklist for Production
*   [ ] **Citation:** No claim without a source link.
*   [ ] **Disclaimer:** "I am an AI, not a lawyer."
*   [ ] **Security:** Encryption at rest and in transit.
