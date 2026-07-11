# Latency Optimization

Achieving a natural, sub-second response time requires tight optimization across the pipeline. 

**Target:** < 1 second voice-to-voice latency.

## Latency Breakdown

Here is a realistic latency budget for our stack:

| Step | Component | Est. Latency |
| :--- | :--- | :--- |
| Audio Capture | Browser AudioWorklet + VAD | ~20 ms |
| Network Upload | WebSockets to Hostinger VPS | ~40 ms |
| STT | Vosk (Local on VPS) | ~80 ms |
| Network (API) | Hostinger VPS to Ollama Cloud | ~60 ms |
| LLM TTFT | Ollama Cloud (Gemma 4 12B) | ~200 ms |
| TTS TTFB | Piper (Local on VPS) | ~150 ms |
| Network Download| Hostinger VPS to Browser | ~40 ms |
| **Total** | | **~590 ms** |

This is highly acceptable for voice interaction (human conversational gap is typically 300-500ms).

## Key Optimizations Implemented

### 1. Streaming the Entire Pipeline
Never wait for a complete response before starting the next step.
- STT outputs partial transcripts.
- LLM outputs streaming tokens.
- TTS processes tokens in chunks.

### 2. Sentence Chunking (The Secret Sauce)
Do not wait for the LLM to finish a whole paragraph.
1. Buffer Ollama Cloud streaming tokens.
2. The moment you detect a sentence boundary (`.`, `?`, `!`), send that sentence to Piper TTS.
3. Stream the audio for Sentence 1 to the browser *while* the LLM generates Sentence 2.

### 3. Client-Side VAD (Voice Activity Detection)
By running Silero VAD in the browser via `onnxruntime-web`, we instantly detect when the user stops speaking. This triggers the LLM call immediately, saving the ~300ms server-side silence detection delay.

### 4. "Filler" Responses
If the LLM call takes longer than expected, the system should instantly play a pre-generated filler phrase to buy time.
- Generate generic phrases (`"Hmm, I see."`, `"That's an interesting point."`) with Piper at startup.
- Cache the PCM bytes in RAM.
- If Ollama Cloud takes > 500ms, push the cached filler PCM to the WebSocket to cover the silence.

### 5. Concise Prompts
The System Prompt explicitly instructs the LLM to:
> "Keep every response under 3 sentences. Be concise."

Shorter LLM responses = less processing time = lower latency.
