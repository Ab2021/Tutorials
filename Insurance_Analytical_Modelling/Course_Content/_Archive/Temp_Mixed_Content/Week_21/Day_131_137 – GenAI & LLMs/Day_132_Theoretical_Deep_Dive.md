# LLMs for Claims Processing (Part 2) - Automation & Adjudication - Theoretical Deep Dive

## Overview
"The average claim takes 10 days to settle. 9 of those days are 'Waiting for Review'."
Claims Processing is the biggest bottleneck in Insurance.
**LLMs (Large Language Models)** can read the First Notice of Loss (FNOL), extract the key facts, and even make a coverage decision in seconds.
Today, we build an **Automated Claims Adjudicator**.

---

## 1. Conceptual Foundation

### 1.1 The Claims Lifecycle

1.  **FNOL (First Notice of Loss):** Customer calls or emails. "I hit a tree."
2.  **Triage:** Is this a Total Loss? Is there Bodily Injury?
3.  **Investigation:** Review Police Report, Photos, Policy.
4.  **Adjudication:** Pay or Deny.
5.  **Settlement:** Send check.

### 1.2 Where LLMs Fit

*   **Extraction:** Reading the Police Report (PDF) and extracting "Driver Name", "Fault", "Date".
*   **Reasoning:** Comparing the "Accident Date" (June 1) to the "Policy Period" (Jan 1 - Dec 31).
*   **Generation:** Drafting the "Reservation of Rights" letter.

---

## 2. Mathematical Framework

### 2.1 Zero-Shot NER (Named Entity Recognition)

*   **Traditional NER:** Trained on specific entities (Person, Org).
*   **LLM NER:** Can extract *anything* you ask for.
    *   *Prompt:* "Extract the 'Cause of Loss' and 'Estimated Damage'."
    *   *Input:* "A pipe burst in the kitchen, ruining the hardwood floor."
    *   *Output:* `{"Cause": "Water Damage", "Estimate": "Unknown"}`.

### 2.2 Semantic Similarity for Fraud

*   **Concept:** Compare the current claim description to 1 million past fraudulent claims.
*   **Embedding:** $v_{\text{claim}} = \text{BERT}(\text{Description})$.
*   **Score:** $\text{FraudScore} = \max(\text{CosineSimilarity}(v_{\text{claim}}, v_{\text{fraud\_db}}))$.
*   *Insight:* Fraudsters reuse stories. If this claim is 99% similar to a known fraud ring's script, flag it.

---

## 3. Theoretical Properties

### 3.1 Chain-of-Thought (CoT) Reasoning

*   **Problem:** LLMs are bad at math and logic if you just ask for the answer.
*   **Solution:** Force the model to "show its work".
*   **Prompt:**
    1.  "What is the Policy Limit?" (\$50k).
    2.  "What is the Claim Amount?" (\$60k).
    3.  "Is Claim > Limit?" (Yes).
    4.  "Therefore, Pay = Limit." (\$50k).

### 3.2 Confidence Calibration

*   **Issue:** LLMs are confident even when wrong.
*   **Technique:** **Self-Consistency**.
    *   Ask the same question 5 times with high temperature.
    *   If the answer is the same 5 times, confidence is High.
    *   If the answer varies, confidence is Low (Flag for Human Review).

---

## 4. Modeling Artifacts & Implementation

### 4.1 The "Adjudicator" Agent (LangChain)

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 1. The Policy (Context)
policy_text = """
Coverage A: Dwelling. Limit: $200,000.
Peril: Fire is Covered. Flood is Excluded.
Deductible: $1,000.
"""

# 2. The Claim (Input)
claim_text = "My house burned down yesterday due to a kitchen fire. Damage is $50,000."

# 3. The Prompt
template = """
You are a Claims Adjuster.
Policy: {policy}
Claim: {claim}

Step 1: Identify the Peril.
Step 2: Check if Peril is Covered.
Step 3: Calculate Payout (Damage - Deductible).
Step 4: Output JSON.
"""

# 4. Execution
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

print(chain.invoke({"policy": policy_text, "claim": claim_text}).content)
# Output:
# {
#   "Peril": "Fire",
#   "Covered": true,
#   "Calculation": "$50,000 - $1,000",
#   "Payout": "$49,000"
# }
```

### 4.2 Image-to-Text (Multimodal)

*   **Input:** Photo of a smashed bumper.
*   **Model:** GPT-4o (Vision).
*   **Prompt:** "Describe the damage severity. Is the car drivable?"
*   **Output:** "Severe damage to front-left bumper and headlight. Potential axle damage. Not drivable."

---

## 5. Evaluation & Validation

### 5.1 The "Golden Set"

*   **Method:** Take 100 closed claims where human adjusters made the correct decision.
*   **Test:** Run the LLM on these 100 claims.
*   **Metric:** Accuracy of "Pay/Deny" decision. Accuracy of "Payout Amount" (within 5%).

### 5.2 Bias Testing

*   **Scenario:** Change the claimant's name from "John Smith" to "Jamal Jones".
*   **Check:** Does the LLM's decision or tone change?
*   **Goal:** Zero variance based on protected attributes.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Context Window" Overflow**
    *   *Scenario:* The Policy Document is 200 pages. It doesn't fit in the prompt.
    *   *Fix:* **RAG (Retrieval Augmented Generation)**. Only retrieve the relevant page (e.g., "Page 45: Fire Coverage").

2.  **Trap: Legal Liability**
    *   *Scenario:* The Chatbot promises "Don't worry, we will pay this."
    *   *Reality:* Coverage is actually excluded.
    *   *Result:* Estoppel. The insurer might be forced to pay because the bot promised it.
    *   *Fix:* Hard-coded guardrails. The bot cannot make binding promises.

---

## 7. Advanced Topics & Extensions

### 7.1 Automated Negotiation

*   **Concept:** Bot negotiates with the Body Shop.
*   **Bot:** "Your estimate for labor is \$150/hr. Market rate is \$120/hr. Can you match?"
*   **Shop:** "Okay, \$130."
*   **Bot:** "Agreed."

### 7.2 Subrogation Mining

*   **Task:** Read the claim to find "Third Party Fault".
*   **Text:** "I was stopped at a red light when the other guy hit me."
*   **Action:** Flag for Subrogation (Sue the other guy's insurance).

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" Adjudication

*   **Regulation:** You must be able to explain *why* a claim was denied.
*   **Solution:** The LLM must output the "Reasoning Trace" (Chain of Thought), citing the specific Policy Clause.

---

## 9. Practical Example

### 9.1 Worked Example: The "Storm Surge"

**Scenario:**
*   **Event:** Hurricane hits Florida. 10,000 claims in 24 hours.
*   **Human Capacity:** 500 claims/day. Backlog = 20 days.
*   **LLM Solution:**
    1.  **Ingest:** All 10,000 FNOLs.
    2.  **Filter:** Identify 2,000 "Simple" claims (e.g., "Fence blown down", < \$5k).
    3.  **Auto-Pay:** LLM verifies coverage, estimates cost, and issues payment via Zelle.
    4.  **Triage:** Route the 8,000 complex claims (e.g., "Roof gone") to humans.
*   **Outcome:** 20% of customers paid in 1 hour. Adjusters focus on the big losses.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Zero-Shot Extraction** turns text into data.
2.  **Chain-of-Thought** ensures logical adjudication.
3.  **Multimodal** models can "see" the damage.

### 10.2 When to Use This Knowledge
*   **Claims VP:** "How do we reduce Loss Adjustment Expense (LAE)?"
*   **Innovation Lead:** "Can we automate the 'Small Claims' desk?"

### 10.3 Critical Success Factors
1.  **Guardrails:** Never auto-deny. Only auto-pay or refer to human.
2.  **Latency:** The bot must respond in seconds, not minutes.

### 10.4 Further Reading
*   **McKinsey:** "Generative AI in Insurance: The Claims Opportunity".

---

## Appendix

### A. Glossary
*   **FNOL:** First Notice of Loss.
*   **NER:** Named Entity Recognition.
*   **CoT:** Chain of Thought.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Similarity** | $\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$ | Fraud Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
