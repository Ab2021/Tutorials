# Large Language Model (LLM) Architectural Analysis (2026 Edition)

This document explores the logic engine of the AI Interviewer. The architecture offloads this workload to Ollama Cloud, creating a reliance on external infrastructure while demanding strict state management to maintain sub-second latency.

## 1. The Cloud Offloading Mandate

Running a 12B parameter conversational LLM locally on a 2 vCPU Hostinger VPS is physically impossible for real-time applications. The architecture leverages **Ollama Cloud** (Pro tier).

### 1.1 Pros of Ollama Cloud Offloading
- **Predictable Cost:** Serverless GPU model based on concurrency, not raw tokens.
- **Speed:** Cloud GPUs (A100s/H100s) provide blistering TTFT (Time-To-First-Token), allowing the TTS sentence chunker to remain saturated.

### 1.2 Cons and Risks
- **Network Dependency:** Introduces a mandatory WAN hop. Network jitter directly causes conversational lag.

## 2. Model Selection: The Dual-Model Approach

### 2.1 The Main Brain: Gemma 4 (12B, Quantized)
- *Pros:* Massive context window, excellent instruction-following, strong logical reasoning for evaluating technical answers.
- *Cons:* Too computationally expensive for basic state-tracking tasks.

### 2.2 The Fast Router: Phi-4-mini (3.8B)
- *Pros:* Ultra-fast routing mechanism. Used for intent classification (e.g., determining if the candidate paused or finished their sentence).

## 3. Empathic Prompting & Non-Verbal Injection (2026 Standard)

A major 2026 advancement in AI voice agents is moving away from static "Wall-of-Text" prompts toward dynamic **Empathic Prompting**. 

The LLM must react to the candidate's non-verbal cues (derived from the client-side WebGPU video analysis) without hallucinating or breaking character.

### 3.1 Metadata Injection
Instead of relying solely on the STT text, the FastAPI backend injects metadata tokens into the LLM's context window. 
For example, if the video analysis detects looking away and frowning, the transcript is passed to the LLM as:
`[METADATA: EMOTION=NERVOUS, ENGAGEMENT=LOW] Candidate: "I think the answer is polymorphism... but I'm not sure."`

### 3.2 State Machine Orchestration
In 2026, relying purely on the LLM to figure out *how* to respond to emotions is inefficient and leads to latency. Instead, the backend uses a state machine.
- If `ENGAGEMENT < 0.4` for 30 seconds, the state machine explicitly prefixes the LLM prompt with: `<SYSTEM_INSTRUCTION> The candidate seems disengaged. Ask a lighter, more interactive follow-up question. </SYSTEM_INSTRUCTION>`.
- *Pros:* This modular "Prompt Blueprint" approach ensures the agent remains deterministic and reliable while exhibiting high emotional intelligence.

## 4. Voice-Specific Prompt Engineering

### 4.1 The "No Markdown" Rule
LLMs default to Markdown (bullet points, bold text). If output to TTS, the TTS engine will try to pronounce the asterisks, causing glitchy audio. The system prompt aggressively enforces plain-text, spoken-word outputs.

### 4.2 Pacing and Conciseness
To prevent the LLM from delivering unnatural, 60-second monologues:
- `num_predict` is capped (e.g., 150 tokens).
- The system prompt enforces 2-3 sentence bursts, ending with a question to seamlessly hand the turn back to the candidate.

## 5. Context Window and Memory Management

### 5.1 Rolling Context Windows
Instead of passing the entire 30-minute interview transcript into every prompt (which destroys TTFT), the backend utilizes a strict rolling window (last 10-20 messages).
- *Pros:* Keeps prompt size small, ensuring instant TTFT.
- *Alternative:* Vector DB (RAG). Rejected for the MVP as it introduces massive complexity for a problem that rolling context handles perfectly for linear interviews.
