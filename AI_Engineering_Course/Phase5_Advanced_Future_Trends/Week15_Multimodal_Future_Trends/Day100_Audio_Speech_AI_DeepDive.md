# Day 100: Audio & Speech AI
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Real-Time Transcription Loop

We will use `faster-whisper` (optimized CTranslate2 implementation) for real-time ASR.

```python
from faster_whisper import WhisperModel
import pyaudio
import numpy as np

# 1. Load Model (Quantized for speed)
model = WhisperModel("small", device="cuda", compute_type="int8")

# 2. Audio Stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

print("Listening...")

while True:
    # 3. Buffer Audio (Simplified)
    data = stream.read(4096)
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 4. Transcribe
    segments, info = model.transcribe(audio_np, beam_size=5)
    
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Text-to-Speech (TTS) Pipeline

Modern TTS (like XTTS) works in two stages:
1.  **Acoustic Model:** Text -> Mel Spectrogram.
2.  **Vocoder:** Mel Spectrogram -> Waveform (HiFi-GAN).

```python
# Pseudo-code for Voice Cloning
speaker_embedding = encoder.encode_speaker("user_voice.wav")
spectrogram = synthesizer.synthesize("Hello world", speaker_embedding)
waveform = vocoder.generate(spectrogram)
```

### Voice Activity Detection (VAD)

You don't want to transcribe silence.
*   **Silero VAD:** A tiny neural network (0.1MB) that runs in 1ms.
*   **Logic:**
    *   If `prob(speech) > 0.5` for 300ms -> Start Segment.
    *   If `prob(speech) < 0.1` for 500ms -> End Segment (User finished sentence).

### Summary

*   **Quantization:** Essential. `int8` Whisper is 4x faster than `fp32` with minimal accuracy loss.
*   **VAD:** The unsung hero of voice interfaces. Without it, the agent interrupts you or waits too long.
