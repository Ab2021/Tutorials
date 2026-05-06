# 🏥 Industry Case Study 1: Healthcare RAG at Scale (Optum / UHG)

## 📌 The Interview Scenario
**Interviewer:** "We are building an internal clinical co-pilot. Doctors use it to query longitudinal patient histories—sometimes spanning 500+ pages of PDFs, clinical notes, and lab results. The system is currently slow (taking 15+ seconds to answer) and occasionally hallucinates medical codes. Furthermore, compliance is terrified about PII leakage. **Walk me through how you would re-architect this system for scale, speed, and safety.**"

---

## 1. Tackling the Latency (TTFT) Problem

**The Trap:** Junior engineers will say "Just switch to a smaller model."
**The Senior Answer:** 
"A 500-page patient history translates to roughly 150,000 tokens. The latency bottleneck here is the **Prefill Phase** of the LLM. Processing 150k tokens autoregressively to build the KV Cache before the first token is generated takes 15 seconds. 

To solve this, I would implement **Provider-Level Prompt Caching**. Because a doctor usually asks 5-10 questions about the *same* patient file in a single session, we can send the patient history as a cached prefix on the first turn. 
- **Turn 1:** 15 seconds TTFT (standard).
- **Turns 2-10:** < 1 second TTFT, because the KV Cache is stored on the LLM's GPUs and the 150k token prefill is skipped.
This also reduces our input token API costs by 80-90% for subsequent queries."

---

## 2. Ensuring Medical Factuality (Anti-Hallucination)

**The Trap:** Junior engineers will say "I'll write a strict system prompt telling it not to hallucinate."
**The Senior Answer:**
"Prompt engineering is probabilistic; we need a deterministic evaluation framework. For clinical data, we cannot rely solely on the LLM.
1. **Automated CI/CD (RAGAS):** I would integrate RAGAS into our deployment pipeline. Specifically, we would track the **Faithfulness** metric to ensure the LLM never introduces medical facts outside of the retrieved context, and **Context Recall** to ensure our Vector DB isn't missing critical lab results.
2. **Contextual Grounding Guardrails:** In AWS Bedrock, I would enable the Contextual Grounding guardrail at runtime. If the LLM generates a response that deviates from the retrieved patient file, the guardrail intercepts it before it reaches the doctor and returns a safe fallback message.
3. **HITL Pipeline:** I would mandate a Human-in-the-Loop sampling pipeline where 2% of generated answers are reviewed by actual clinicians to evaluate clinical safety nuances that an LLM-as-a-judge might miss."

---

## 3. Handling Strict PII / PHI Compliance (HIPAA)

**The Trap:** "I'll write a python regex script to redact SSNs before sending to the model."
**The Senior Answer:**
"Custom regex is brittle and difficult to maintain for complex PHI (Protected Health Information). I would leverage **Amazon Bedrock Guardrails for Sensitive Information**. 
- We configure the guardrail to actively detect PII (names, SSNs, medical record numbers).
- We set the policy to **Mask** (`***`). 
- **Architecture:** The raw text hits the Bedrock API boundary. The Guardrail evaluates and masks the text *before* the prompt is processed by the foundational model. This ensures no PHI is ever processed by the LLM's core weights or logged in standard inference logs, preserving HIPAA compliance without maintaining custom redaction lambdas."

---

## 4. Addressing VRAM Constraints (If self-hosting)

**Interviewer Follow-up:** "What if we can't use Bedrock and must self-host open-source Llama-3 locally due to strict data residency? How do you handle the 150k token context window on our limited GPU cluster?"

**The Senior Answer:**
"Self-hosting 150k tokens per request will immediately trigger Out-Of-Memory (OOM) errors because the KV Cache size scales linearly with sequence length.
1. **Inference Engine:** I would deploy the model using **vLLM** to take advantage of **PagedAttention**. This eliminates memory fragmentation by storing the KV Cache in non-contiguous physical blocks, freeing up massive VRAM.
2. **Quantization:** If we still lack VRAM, I would apply **FP8 KV Cache Quantization**. By compressing the KV tensors from 16-bit to 8-bit, we instantly double the available context window size per GPU with minimal impact on retrieval accuracy."

---

## 💡 Key Takeaways for Healthcare Interviews
- Emphasize **deterministic safety** over probabilistic prompting.
- Always address the **Prefill Bottleneck** when dealing with "Patient Histories" or "Longitudinal Records".
- Know the difference between **Faithfulness** (did the LLM lie?) vs **Context Precision** (did the search engine fail?).
