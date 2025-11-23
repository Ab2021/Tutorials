# Day 97: Healthcare & Medical Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between "De-identification" and "Anonymization"?

**Answer:**
*   **De-identification:** Removing specific identifiers (Name, SSN) defined by HIPAA Safe Harbor. Re-identification might still be possible.
*   **Anonymization:** Irreversibly destroying links to the individual.
*   For LLMs, we usually de-identify before sending to the cloud if we don't have a BAA.

#### Q2: Why is "Explainability" crucial in CDSS?

**Answer:**
Black box: "Diagnose Cancer."
Doctor: "Why?"
Black box: "I don't know."
*   This is unacceptable. The model must cite the specific pixels in the X-Ray or the specific lab values that led to the decision.

#### Q3: How do you handle "Bias" in medical AI?

**Answer:**
Training data often under-represents minorities.
*   **Result:** Skin cancer models fail on dark skin. Pulse oximeters fail on dark skin.
*   **Fix:** Balanced datasets, stratified evaluation (test performance on each subgroup separately).

#### Q4: What is "Drift" in healthcare?

**Answer:**
Medical practice changes.
*   **Example:** COVID-19 appeared. Models trained in 2019 didn't know about it.
*   **Fix:** Continuous monitoring and retraining.

### Production Challenges

#### Challenge 1: The "Silent Failure"

**Scenario:** The Scribe misses a "Not".
Transcript: "Patient does NOT have allergies."
Note: "Patient has allergies."
**Root Cause:** Attention failure.
**Solution:**
*   **Verbal Verification:** Doctor reviews the note before signing.
*   **Highlighting:** Highlight the source text corresponding to the generated sentence.

#### Challenge 2: Integration with EHR (Epic/Cerner)

**Scenario:** You built a great bot, but it's a separate tab. Doctors won't use it.
**Root Cause:** Workflow friction.
**Solution:**
*   **SMART on FHIR:** The standard for integrating apps into EHRs. The bot must live *inside* Epic.

#### Challenge 3: Liability

**Scenario:** Bot suggests wrong dose. Patient harmed. Who is sued?
**Root Cause:** Legal ambiguity.
**Solution:**
*   **Human in the Loop:** The Doctor is the "Learned Intermediary". The AI is just a tool. The Doctor takes full responsibility for the final decision.

### System Design Scenario: Telehealth Triage Bot

**Requirement:** Screen patients before video call.
**Design:**
1.  **Safety Layer:** Check for "Red Flags" (Chest pain, difficulty breathing). If yes -> "Call 911" -> End Chat.
2.  **History Taking:** Ask standard questions (Onset, Duration, Severity).
3.  **Summary:** Present a 3-bullet summary to the doctor before they join the call.
4.  **Privacy:** Ephemeral storage. Delete chat after call ends.

### Summary Checklist for Production
*   [ ] **BAA:** Signed with OpenAI/AWS.
*   [ ] **Audit Logs:** Who accessed whose record and when?
*   [ ] **Disclaimer:** "This is not medical advice."
