# Day 43: NLP for Robotics (Voice Commands)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> Humans don't speak Binary. Robots must speak English.
> - **Focus:** Speech-to-Text (Whisper), Intent Classification (LLMs), and Slot Filling.
> - **Code:** A Voice-Commanded Robot ("Go to the kitchen") using OpenAI Whisper and a simple Intent Parser.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** Automatic Speech Recognition (ASR) into a ROS 2 node.
2.  **Train** a simple NLU (Natural Language Understanding) model to map sentences to Actions + Parameters.
3.  **Implement** a Keyword Spotting system ("Hey Robot") for Wake-Word detection.
4.  **Connect** NLP outputs to the Navigation Stack (Waypoints).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Microphone.

### Software Environment
```bash
pip install openai-whisper sounddevice scipy
pip install transformers torch
```

### Prior Knowledge
- Python State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pipeline

Voice $\to$ Text $\to$ Intent $\to$ Action.
1.  **VAD (Voice Activity Detection):** Is someone speaking? (Ignore silence).
2.  **ASR (Speech-to-Text):** Convert audio wave to string ("Go to the kithen").
3.  **NLU (Understanding):**
    *   **Intent:** `NAVIGATE`
    *   **Slots:** `Target=Kitchen`
4.  **Planner:** `move_base(map['Kitchen'])`.

### 🔹 Part 2: Modern ASR (Whisper)

OpenAI Whisper is a Transformer trained on 680k hours of audio.
*   *Pros:* Extremely robust to accents and noise. Works Offline (Tiny/Base models).
*   *Cons:* High latency on CPU (needs optimization).

### 🔹 Part 3: Intent Recognition (Zero-Shot)

Instead of training a custom classifier (Rasa/Dialogflow), we can use Large Language Models (LLMs) or Zero-Shot classifiers (BART-NLI).
*   Prompt: "Extract command from: 'Fetch a beer from the fridge'."
*   LLM Output: `{ "action": "FETCH", "object": "beer", "location": "fridge" }`.

---

## 💻 Implementation: "Hey Robot"

We will build a complete voice command node.

### 🛠️ Project Structure
```text
day43_nlp/
├── src/
│   ├── audio_capture.py
│   ├── whisper_node.py
│   └── command_parser.py
└── run_voice_ctrl.py
```

### 👨‍💻 Audio Capture (`src/audio_capture.py`)

```python
import sounddevice as sd
import numpy as np
import queue

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_audio(duration=5, fs=16000):
    print("Listening...")
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        sd.sleep(int(duration * 1000))
    
    # Concatenate all blocks
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
    return np.concatenate(audio_data, axis=0)
```

### 👨‍💻 Whisper Node (`src/whisper_node.py`)

```python
import whisper
import torch
import numpy as np

class SpeechRecognizer:
    def __init__(self, model_size="tiny"):
        print(f"Loading Whisper {model_size}...")
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_array):
        # Whisper expects float32 [-1, 1], 16kHz
        audio_array = audio_array.flatten().astype(np.float32)
        
        result = self.model.transcribe(audio_array, fp16=False)
        text = result['text'].strip()
        print(f"Heard: '{text}'")
        return text
```

### 👨‍💻 Command Parser (`src/command_parser.py`)

Simple regex/keyword based parser (Fast, Offline).

```python
import re

class IntentParser:
    def __init__(self):
        self.locations = {
            "kitchen": [10, 5],
            "bedroom": [2, 2],
            "lab": [0, 0]
        }
    
    def pars(self, text):
        text = text.lower()
        
        # Rule 1: Navigation
        if "go to" in text or "navigate to" in text:
            for loc in self.locations:
                if loc in text:
                    return {"intent": "NAVIGATE", "target": loc, "coords": self.locations[loc]}
                    
        # Rule 2: Stop
        if "stop" in text or "halt" in text:
            return {"intent": "STOP"}
            
        return {"intent": "UNKNOWN"}
```

### 👨‍💻 Main Loop (`run_voice_ctrl.py`)

```python
from src.audio_capture import record_audio
from src.whisper_node import SpeechRecognizer
from src.command_parser import IntentParser

asr = SpeechRecognizer()
nlu = IntentParser()

while True:
    input("Press Enter to speak...")
    audio = record_audio(duration=3)
    
    text = asr.transcribe(audio)
    
    cmd = nlu.pars(text)
    print(f"Executing: {cmd}")
    
    if cmd['intent'] == "NAVIGATE":
        print(f"Sending goal: {cmd['coords']}")
        # publish_to_ros(cmd['coords'])
```

---

## 🔬 Lab Exercise: The "Echo" Problem

### 1. Lab Objectives
- Run the robot. Play the robot's TTS (Text-To-Speech) saying "I am going to the kitchen".
- **Problem:** The microphone hears the robot's own speaker. It transcribes "I am going to the kitchen", parses it as a command, and loops forever.
- **Fix:** **AEC (Acoustic Echo Cancellation)**.
    - Software: `WebRTC` AEC.
    - Hardware: ReSpeaker Mic Array (DSP subtraction).
    - Hack: Mute Mic while TTS is playing.

---

## 🚀 Project: "LLM Agent"

**Goal:** Connect ChatGPT/Llama 2 to the robot.
1.  **Prompt:** "You are a Robot Operating System interface. User says: 'I'm thirsty'. Available tools: [navigate(loc), find_object(obj), pickup(obj)]. Return JSON plan."
2.  **Input:** "I'm thirsty."
3.  **Output:** `[ {"tool": "find_object", "args": "water_bottle"}, {"tool": "pickup", "args": "water_bottle"}, {"tool": "navigate", "args": "user"} ]`.
4.  **Result:** High-level reasoning ("Thirsty" -> "Get Water").

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Hallucination"
*   **Symptom:** Silence is transcribed as "Thank you for watching" (common YouTube credit Whisper trained on).
*   **Fix:** Check `avg_logprob`. If confidence is low, discard. Use VAD to ensure speech actually exists before transcribing.

#### 2. "Latency"
*   **Symptom:** Robot reacts 5 seconds later. User thinks it didn't hear.
*   **Fix:** **Streaming ASR**. Transcribe chunk-by-chunk. Show "Thinking..." LED immediately.

---

## ⚡ Optimization: Quantized Whisper

Run `whisper.cpp` (C++ port).
*   Optimized with AVX instructions and INT8 quantization.
*   Runs "Base" model in real-time on a Raspberry Pi 4 CPU.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Wake Word"?
    *   **A:** A lightweight model (running always) that listens for "Keyword". Triggering it wakes up the heavy ASR model. Saves privacy and battery.
2.  **Q:** Why not just string matching?
    *   **A:** "Go to kitchen", "Drive to kitchen", "Head to kitchen". String match fails. NLU captures semantic meaning.
3.  **Q:** What is "Slot Filling"?
    *   **A:** Extracting specific parameters (Entities) from the intent. Intent=Move, Slot=Destination.

### Challenge Task
> **Task:** Context Awareness.
> 1. User: "Go to the kitchen." -> Robot goes.
> 2. User: "Go back."
> 3. Robot must remember "Back" means "From Kitchen to Previous Loc". Requires **Dialogue State Tracking**.

---

## 📚 Further Reading
- **OpenAI Whisper:** GitHub.
- **Rasa NLU:** Open source conversational AI.

---

**Day 43 Complete**
