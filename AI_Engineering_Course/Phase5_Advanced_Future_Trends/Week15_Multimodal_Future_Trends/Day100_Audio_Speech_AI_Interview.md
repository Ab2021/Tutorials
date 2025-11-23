# Day 100: Audio & Speech AI
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does Whisper handle multiple languages?

**Answer:**
It uses a `<language>` token at the start of the sequence.
*   Input: `[<|startoftranscript|>, <|es|>, <|transcribe|>]` + Audio.
*   Output: Spanish text.
*   It was trained on a massive mix of languages, allowing it to learn cross-lingual representations.

#### Q2: What is the "Cocktail Party Problem"?

**Answer:**
Isolating one speaker from a noisy room.
*   **Solution:** Speaker Diarization (Clustering audio embeddings).
*   **Directional Audio:** Using microphone arrays to beamform.

#### Q3: How do you reduce TTS latency?

**Answer:**
*   **Streaming:** Start playing the audio as soon as the first chunk of spectrogram is generated. Don't wait for the whole sentence.
*   **Non-Autoregressive Models:** FastSpeech2 generates the whole spectrogram in parallel (fast), unlike Tacotron which is autoregressive (slow).

#### Q4: What are the security risks of Voice Cloning?

**Answer:**
*   **Vishing (Voice Phishing):** Cloning a CEO's voice to authorize a transfer.
*   **Defense:** Audio Watermarking (inaudible noise added to generated audio) and Liveness Detection.

### Production Challenges

#### Challenge 1: Hallucination in Whisper

**Scenario:** In a silent room, Whisper outputs "Thank you for watching."
**Root Cause:** Training data bias. Many YouTube videos end with "Thank you for watching," and Whisper hallucinates this pattern in silence.
**Solution:**
*   **VAD:** Aggressive filtering of silence.
*   **Prompting:** Provide a prompt like "Silence." to bias the model.

#### Challenge 2: Latency Stacking

**Scenario:** ASR (1s) + LLM (2s) + TTS (1s) = 4s delay. User hangs up.
**Root Cause:** Sequential processing.
**Solution:**
*   **Full Duplex Streaming:** ASR streams tokens to LLM. LLM streams tokens to TTS. TTS streams audio to User. Overlap everything.

#### Challenge 3: Accents and Jargon

**Scenario:** Whisper fails on "Hydrochlorothiazide" spoken with a heavy accent.
**Root Cause:** Out of distribution.
**Solution:**
*   **Prompting:** Pass a list of expected keywords (e.g., drug names) in the `initial_prompt` to Whisper. It biases the decoder towards those words.

### System Design Scenario: AI Call Center Agent

**Requirement:** Replace Tier-1 support.
**Design:**
1.  **Telephony:** Twilio Media Streams (WebSocket).
2.  **VAD:** Silero.
3.  **ASR:** Deepgram (Faster than Whisper).
4.  **Brain:** GPT-4o (Multimodal Audio native) or Llama-3.
5.  **TTS:** ElevenLabs Turbo.
6.  **Latency Budget:** 800ms total.

### Summary Checklist for Production
*   [ ] **Fallback:** If ASR confidence is low, ask "Could you repeat that?"
*   [ ] **Barge-In:** Allow the user to interrupt the bot (cancel TTS playback when VAD detects speech).
