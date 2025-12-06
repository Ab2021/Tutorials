# Day 179: Natural Language Instruction
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> "Robot, fetch me a beer."
> - **Focus:** Speech Recognition (Whisper), Large Language Models (LLMs), Grounding (Text $\to$ Coordinates), and Prompt Engineering for Robotics.
> - **Code:** `voice_agent.py`. A standard pipeline: Audio $\to$ Text $\to$ Intent $\to$ Action.
> - **Note:** Discuss the "Symbol Symbol" problem (Grounding).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** Whisper (OpenAI) for robust Speech-to-Text.
2.  **Design** a Prompt Structure to convert Natural Language into JSON Commands.
3.  **Implement** a Semantic Map lookup ("Where is the kitchen?").
4.  **Handle** Ambiguity ("Which cup?").

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Microphone.
- Speaker (TTS).

### Software Environment
```bash
pip install openai-whisper sounddevice numpy requests
```

### Prior Knowledge
- REST APIs.
- Semantic Mapping (Week 17).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Pipeline

1.  **ASR (Automatic Speech Recognition):** Audio Wave $\to$ "Go to the kitchen". (Whisper).
2.  **NLU (Natural Language Understanding):** Text $\to$ Intent: `NAVIGATE`, Entity: `KITCHEN`. (LLM/BERT).
3.  **Grounding:** `KITCHEN` $\to$ `(x=5.0, y=2.0)`. (Database).
4.  **Execution:** Send Goal to Nav2.

### 🔹 Part 2: The Grounding Problem

Words are abstract symbols. Robots live in metric space.
*   **"Pick up the red block"**:
    *   Vision System must output: `[ {id:1, color:red, pos:x}, {id:2, color:blue} ]`.
    *   Language Agent maps "red block" $\to$ `id:1`.

### 🔹 Part 3: LLMs in Robotics

*   **Zero-Shot Planning:** "I spilled my drink."
    *   LLM Output:
        1.  Find Sponge.
        2.  Pick up Sponge.
        3.  Go to Spil.
        4.  Wipe.
*   **Issues:** LLMs hallucinate. They assume you have a sponge.

---

## 💻 Implementation: The Voice Commander

We use a local mock for the LLM (or regex) to avoid API costs in learning, but the structure assumes an LLM call.

### 🛠️ Project Structure
```text
day179_nlp/
├── src/
│   ├── voice_agent.py
│   ├── semantic_db.json
│   └── mock_whisper.py
└── README.md
```

### 👨‍💻 Semantic Database (`src/semantic_db.json`)

```json
{
    "kitchen": {"x": 5.0, "y": 2.0},
    "living_room": {"x": 0.0, "y": 0.0},
    "charger": {"x": -2.0, "y": -2.0},
    "red_cup": {"x": 5.1, "y": 2.1, "type": "object"}
}
```

### 👨‍💻 Voice Agent (`src/voice_agent.py`)

```python
import json
import re
import time
import math

class RobotExecutor:
    def __init__(self):
        with open('src/semantic_db.json', 'r') as f:
            self.map = json.load(f)
            
        self.current_pos = [0,0]
        
    def navigate(self, target_name):
        if target_name in self.map:
            coords = self.map[target_name]
            print(f"[ROBOT] Navigating to {target_name} at {coords}...")
            time.sleep(1)
            self.current_pos = [coords['x'], coords['y']]
            print("[ROBOT] Arrived.")
            return True
        else:
            print(f"[ROBOT] ERROR: I don't know where '{target_name}' is.")
            return False
            
    def pickup(self, obj_name):
        if obj_name in self.map:
            # Check range
            r_x, r_y = self.current_pos
            o_x, o_y = self.map[obj_name]['x'], self.map[obj_name].get('y', 0)
            dist = math.sqrt((r_x-o_x)**2 + (r_y-o_y)**2)
            
            if dist < 0.5:
                print(f"[ROBOT] Picking up {obj_name}.")
                return True
            else:
                print(f"[ROBOT] Too far from {obj_name} ({dist:.1f}m). Navigate there first.")
                return False
        return False

class NLPAgent:
    def __init__(self):
        self.executor = RobotExecutor()
        
    def parse_command(self, text):
        """
        Mock LLM. Translates text to (Intent, Entity).
        """
        text = text.lower()
        
        # Regex heuristics for demo
        if "go to" in text or "navigate" in text:
            # Extract "the X"
            words = text.split()
            # Find word after "to"
            try:
                idx = words.index("to")
                target = words[idx+1]
                if target == "the": target = words[idx+2]
                return "NAVIGATE", target.strip(".,!?")
            except:
                return "UNKNOWN", None
                
        elif "pick up" in text:
            if "red cup" in text: return "PICKUP", "red_cup"
            
        return "UNKNOWN", None
        
    def run_pipeline(self, audio_mock_text):
        print(f"\nUser said: '{audio_mock_text}'")
        
        # 1. NLU
        intent, entity = self.parse_command(audio_mock_text)
        print(f"  -> Parsed: {intent} on {entity}")
        
        # 2. Execution
        if intent == "NAVIGATE":
            self.executor.navigate(entity)
        elif intent == "PICKUP":
            self.executor.pickup(entity)
        else:
            print("  -> I didn't understand that.")

def main():
    agent = NLPAgent()
    
    # Trace 1
    agent.run_pipeline("Robot, go to the kitchen.")
    
    # Trace 2
    agent.run_pipeline("Please pick up the red cup.")
    
    # Trace 3 (Fail)
    agent.run_pipeline("Go to Mars.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "LLM API Integration"

### 1. Lab Objectives
- **Key API:** Get an OpenAI Key (or run local Llama2 via Ollama).
- **Prompt:**
    ```text
    You are a robot. 
    Available actions: [navigate(location), pickup(object)].
    Map: [kitchen, bedroom].
    User: "I'm hungry."
    Response (JSON): {"action": "navigate", "arg": "kitchen"}
    ```
- **Integrate:** Replace the regex parser in `voice_agent.py` with this API call.
- **Test:** Say "I'm hungry" and watch the robot go to the kitchen.

---

## 🚀 Project: "Voice-Controlled Drone"

**Goal:** Natural Language Flight.
1.  **Commands:** "Take off", "Fly up 2 meters", "Land".
2.  **Safety:** If user says "Do a barrel roll", LLM might say OK, but Safety Layer (Day 173) must block it.
3.  **Implementation:** Whisper Micro on Edge or Cloud API.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Ambiguous Entities"
*   **User:** "Go to the charger."
*   **Problem:** There are 5 chargers.
*   **Fix:** Disambiguation Dialogue. Robot: "Which charger? Living room or Bedroom?".

#### 2. "Latency"
*   **Cause:** Whisper Large + GPT-4 = 5 seconds delay.
*   **Fix:** Use Faster-Whisper (INT8) and smaller models (Phi-2, Mistral 7B). Provide "Thinking sound" (Beep boop) to user so they know you heard them.

---

## ⚡ Optimization: Keyword Spotting (Wake Word)

Don't run Whisper 24/7. Privacy + Battery.
*   **Wake Word:** "Hey Robot" (Porcupine / OpenWakeWord).
*   **Pipeline:**
    1.  Low power MCU listens for "Hey Robot".
    2.  If detected, wake up GPU.
    3.  Record Audio.
    4.  Send to Whisper.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Grounding?
    *   **A:** The process of mapping abstract symbols (words) to physical perceptions (sensor data) or actions.
2.  **Q:** Why use JSON for LLM output?
    *   **A:** Deterministic parsing. It's easier to code `json.loads(response)` than to parse free text paragraphs.

### Challenge Task
> **Task:** Spatial Prepositions.
> 1. "Go *near* the kitchen." (Not *in* it).
> 2. "Go *between* the table and the chair."
> 3. Compute target coords based on geometric relationships.

---

## 📚 Further Reading
- **SayCan (Google):** Grounding Language in Robotic Affordances.
- **OpenAI Whisper:** Sota ASR.

---

**Day 179 Complete**
