# Day 97: Healthcare & Medical Agents
## Core Concepts & Theory

### The Critical Domain

Healthcare AI saves lives but carries the highest risk.
*   **Applications:** Scribing, Diagnosis Support, Patient Triage, Drug Discovery.
*   **Constraint:** HIPAA (Privacy) and FDA (Safety).

### 1. Medical Scribing (Ambient Intelligence)

Doctors spend 50% of their time typing notes.
*   **Task:** Listen to doctor-patient conversation -> Generate SOAP Note (Subjective, Objective, Assessment, Plan).
*   **Tech:** Whisper (ASR) + LLM (Summarization).
*   **Challenge:** Medical jargon, accents, speaker diarization.

### 2. Clinical Decision Support (CDSS)

"Doctor, have you considered Lupus?"
*   **Differential Diagnosis:** Agent analyzes symptoms + history -> Suggests list of possible conditions.
*   **Interaction Check:** "Patient is taking Warfarin. Don't prescribe Aspirin."
*   **Evidence Retrieval:** Fetching relevant papers from PubMed.

### 3. Patient Triage (Chatbots)

"I have a headache."
*   **Symptom Checker:** Ask follow-up questions ("Is it throbbing?", "Do you have fever?").
*   **Triage:** "Call 911" vs "See GP" vs "Take Tylenol".
*   **Safety:** Must err on the side of caution.

### 4. Privacy (HIPAA)

*   **PHI (Protected Health Information):** Name, DOB, MRN.
*   **De-identification:** Removing PHI before processing (if possible).
*   **BAA (Business Associate Agreement):** Legal contract required to send PHI to a cloud provider.

### Summary

Medical Agents reduce burnout for doctors and improve access for patients. But they must be **Assistants**, not **Replacements**. The human MD always has the final say.
