# Day 79: Specialized Domain Agents (Code, Legal, Medical)
## Core Concepts & Theory

### Domain Adaptation

**General Models:** Good at generic tasks.
**Domain Models:** Need specific vocabulary, reasoning patterns, and tools.
- **Code:** Syntax, Libraries, Debugging.
- **Legal:** Precedents, Citations, Logic.
- **Medical:** Anatomy, Diagnosis, Privacy.

### 1. Coding Agents (The "Devin" Pattern)

**Capabilities:**
- **Read:** Parse codebase (AST).
- **Plan:** Break down feature request.
- **Edit:** Modify files.
- **Run:** Execute tests and fix errors.

**Tools:**
- **LSP (Language Server Protocol):** For autocomplete/definitions.
- **Terminal:** For running commands.
- **Git:** For version control.

**Challenges:**
- **Context:** Codebases are huge (millions of lines). Need RAG or Repo Map.
- **Dependency:** Changing one file breaks another.

### 2. Legal Agents

**Capabilities:**
- **Discovery:** Finding relevant documents in TBs of emails.
- **Drafting:** Writing contracts.
- **Analysis:** "Does this clause violate GDPR?"

**Challenges:**
- **Hallucination:** Citing fake cases (Sanctions!).
- **Precision:** Every word matters.
- **Privacy:** Client privilege.

### 3. Medical Agents

**Capabilities:**
- **Scribe:** Transcribing doctor-patient notes.
- **Diagnosis Support:** Suggesting differentials.
- **Triage:** "Is this urgent?"

**Challenges:**
- **Safety:** Wrong advice kills.
- **Bias:** Training data bias.
- **HIPAA:** Strict data handling.

### 4. Repo-Level Context (Code)

**Techniques:**
- **Repo Map:** A compressed tree structure of the codebase (file names, class names).
- **Dependency Graph:** Knowing that `auth.py` imports `user.py`.
- **RAG:** Indexing code chunks.

### 5. Med-PaLM / BioMistral

**Models:**
- Models fine-tuned on PubMed, medical textbooks.
- **Evaluation:** USMLE (Medical Licensing Exam).

### 6. LegalBench

**Benchmarks:**
- Evaluating legal reasoning (IRAC - Issue, Rule, Analysis, Conclusion).

### 7. Human-in-the-Loop (Critical)

- **Code:** Dev reviews PR.
- **Legal:** Lawyer reviews contract.
- **Medical:** Doctor signs off on diagnosis.
- **Role:** The agent is a *Copilot*, not an Autopilot.

### 8. Summary

**Domain Strategy:**
1.  **Code:** Use **Repo Maps** and **LSP** tools.
2.  **Legal:** Focus on **Retrieval Precision** and **Citations**.
3.  **Medical:** Focus on **Safety** and **Summarization**.
4.  **Model:** Use **Domain-Specific Fine-tunes** (DeepSeek-Coder, Med-PaLM).
5.  **Review:** Always require **Expert Review**.

### Next Steps
In the Deep Dive, we will implement a Simple Coding Agent with a file system tool, and a Legal Citation Checker.
