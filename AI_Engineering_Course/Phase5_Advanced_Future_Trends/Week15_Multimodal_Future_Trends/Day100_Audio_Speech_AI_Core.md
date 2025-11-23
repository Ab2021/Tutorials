# Day 100: Audio & Speech AI
## Core Concepts & Theory

### The Voice Interface

Voice is the most natural interface.
*   **ASR (Automatic Speech Recognition):** Audio -> Text (Whisper).
*   **TTS (Text-to-Speech):** Text -> Audio (ElevenLabs, XTTS).
*   **Audio Generation:** Text -> Music/SFX (Suno, AudioLDM).

### 1. Whisper (The Transformer for Audio)

OpenAI's Whisper changed everything.
*   **Architecture:** Encoder-Decoder Transformer.
*   **Input:** Log-Mel Spectrogram (Visual representation of sound).
*   **Training:** 680,000 hours of weakly supervised web audio.
*   **Capabilities:** Multilingual, Translation, Timestamping.

### 2. TTS & Voice Cloning

*   **Concatenative (Old):** Gluing recorded snippets. Robotic.
*   **Parametric (Old):** Predicting vocoder parameters. Smooth but buzzy.
*   **Neural (New):** End-to-End Deep Learning.
    *   **VALL-E:** Language modeling for audio tokens.
    *   **Diffusion:** Diffusing noise into a spectrogram.

### 3. Audio Embeddings (CLAP)

Contrastive Language-Audio Pre-training.
*   Like CLIP, but for Audio.
*   Allows searching for "Sad piano music" in a database of MP3s.

### 4. Latency (The Real-Time Constraint)

For a conversation, latency must be < 500ms.
*   **VAD (Voice Activity Detection):** Knowing when the user stopped speaking.
*   **Streaming:** Generating audio chunks while the text is still being generated.

### Summary

Audio AI is solving the "Input/Output" bottleneck for agents. It turns a Chatbot into a Voice Assistant.
