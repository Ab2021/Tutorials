# Day 105: Capstone: Building a Multimodal Future
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you architect a system for < 500ms voice-to-voice latency?

**Answer:**
*   **VAD:** On-device (10ms).
*   **ASR:** Streaming (Deepgram/Whisper). Send partials.
*   **LLM:** Streaming. Start generating after 5 words of ASR.
*   **TTS:** Streaming. Start playing audio after 1 sentence of LLM output.
*   **Network:** WebSockets / gRPC. No HTTP polling.

#### Q2: What is "World Model"?

**Answer:**
An internal simulation of the world.
*   If I drop a glass, it breaks.
*   Video generation models (Sora) are learning World Models.
*   Agents need World Models to plan (predicting the outcome of actions).

#### Q3: How do you handle "Interruption"?

**Answer:**
*   **Full Duplex:** The microphone is always open.
*   **Echo Cancellation:** The system must subtract its own voice (TTS) from the microphone input so it doesn't hear itself.
*   **Logic:** If VAD detects user speech during TTS playback -> Stop TTS immediately -> Clear buffers -> Process new input.

### Production Challenges

#### Challenge 1: The "Hotword" False Positive

**Scenario:** "Hey Jarvis" triggers randomly.
**Root Cause:** Low threshold.
**Solution:**
*   **Two-Stage Wake Word:**
    1.  Low-power DSP (High recall, low precision).
    2.  Verify with larger model (High precision).

#### Challenge 2: Ambient Noise

**Scenario:** User is in a cafe.
**Root Cause:** SNR (Signal-to-Noise Ratio).
**Solution:**
*   **Beamforming:** Use multiple microphones to focus on the user's mouth.
*   **Voice Isolation:** AI model to suppress background noise (e.g., RNNoise).

#### Challenge 3: Cost at Scale

**Scenario:** Multimodal inputs are expensive (Image + Audio + Text).
**Root Cause:** Token volume.
**Solution:**
*   **Early Exit:** If the user just says "Stop", don't send the image to GPT-4o. Use a small text model.

### System Design Scenario: The "Her" OS

**Requirement:** An always-on companion.
**Design:**
1.  **Memory:** Vector DB stores every conversation forever.
2.  **Personality:** Fine-tuned Llama-3-70B for EQ (Emotional Intelligence).
3.  **Vision:** Periodic snapshots (every 10s) to understand context ("Oh, you're eating lunch?").
4.  **Privacy:** All processing on-device (Apple Intelligence approach) or Private Cloud.

### Summary Checklist for Production
*   [ ] **Latency:** Measure "Voice-to-Ear" latency. Target 500ms.
*   [ ] **Safety:** Ensure the agent cannot be tricked into buying things or deleting files.
*   [ ] **Fun:** The most important metric for a companion agent.
