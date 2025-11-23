# Lab 4: Audio Transcription Pipeline

## Objective
Long-form audio transcription.
Whisper has a limit (30s). We need to chunk.

## 1. The Pipeline (`transcribe.py`)

```python
# Mock Whisper
def transcribe_chunk(audio_chunk):
    return "This is a chunk of text."

# 1. Load Audio
audio_len = 100 # seconds
chunk_size = 30

# 2. Loop
full_text = ""
for i in range(0, audio_len, chunk_size):
    print(f"Processing chunk {i}-{i+chunk_size}s")
    text = transcribe_chunk(None)
    full_text += text + " "

print(f"Full Transcript: {full_text}")
```

## 2. Challenge
Implement **Speaker Diarization** using `pyannote.audio`.
Output: "Speaker A: Hello. Speaker B: Hi."

## 3. Submission
Submit the diarized transcript.
