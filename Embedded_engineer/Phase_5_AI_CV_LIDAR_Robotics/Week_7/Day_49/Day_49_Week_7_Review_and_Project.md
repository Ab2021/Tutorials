# Day 49: Week 7 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> We made robots smart (AI) and fast (Optimized). Now we made them polite.
> - **Goal:** A full HRI stack for a Receptionist Robot.
> - **Code:** Integration of Whisper (Voice), MediaPipe (Gesture), DeepFace (Emotion), and Safety Monitors.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** multiple AI modalities (Voice, Vision, Emotion) into a State Machine.
2.  **Architect** a "Human-in-the-Loop" validation system (Robot asks for clarification).
3.  **Deploy** a Safety Monitor that overrides social behaviors (Etiquette < Safety).
4.  **Demonstrate** a fully interactive "Tour Guide" scenario.

---

## 📚 Week 7 Review: The HRI Layer

| Day | Topic | Technology | Interaction |
|-----|-------|------------|-------------|
| **43** | **NLP** | Whisper / LLMs | "Go to the kitchen" |
| **44** | **Gesture** | MediaPipe | Pointing, Stop Hand |
| **45** | **Intent** | Social LSTM | Yielding to walkers |
| **46** | **Safety** | ISO 13482 | E-Stop, Speed Clamping |
| **47** | **XAI** | Grad-CAM | "Why did you stop?" |
| **48** | **Emotion** | DeepFace | Mirroring, Empathy |

### The "Social" Robot Architecture
1.  **Perception:** See Face, Hear Voice, Sense Obstacles.
2.  **Cognition:** Parse Intent, Estimate Emotion, Plan Path.
3.  **Action:** Speak Response, Move Base, Display Face.

---

## 🚀 Weekly Capstone: "The AI Receptionist"

**Scenario:** A robot at a hotel lobby.
**Workflow:**
1.  **Detect Guest:** Face Detection (Wake up).
2.  **Estimate Mood:** If Angry $\to$ Apologize. If Happy $\to$ Joke.
3.  **Listen:** "Where is the gym?"
4.  **Gesture:** Robot points right arm + Moves head.
5.  **Guide:** "Follow me" (Check safe distance).

### 🛠️ Project Structure
```text
week7_capstone/
├── src/
│   ├── perception_node.py (Voice+Vision)
│   ├── dialogue_manager.py (State Machine)
│   ├── safety_layer.py
│   └── main_brain.py
└── run_receptionist.py
```

### 👨‍💻 Component 1: Perception Fusion (`src/perception_node.py`)

Runs threads for Mic and Cam. Fuses data into a `UserFrame`.

```python
class PerceptionEngine:
    def get_user_state(self):
        # 1. Vision
        img = self.cam.read()
        landmarks = self.hands.process(img)
        emotion = self.face.analyze(img)
        
        # 2. Audio
        audio = self.mic.record(1.0) # Short buffer
        text = self.whisper.transcribe(audio)
        
        return {
            "hand_gesture": self.classify_gesture(landmarks),
            "face_emotion": emotion,
            "speech_text": text
        }
```

### 👨‍💻 Component 2: Dialogue State Machine (`src/dialogue_manager.py`)

Using `transition` library or simple dict.

```python
class ReceptionistBrain:
    def __init__(self):
        self.state = "IDLE"
        
    def update(self, perception):
        if self.state == "IDLE":
            if perception['face_emotion']: 
                self.state = "GREETING"
                
        elif self.state == "GREETING":
            mood = perception['face_emotion']
            if mood == 'happy':
                print("Robot: Welcome! Great to see you smiling!")
            else:
                print("Robot: Hello. How can I assist you today?")
            self.state = "LISTENING"
            
        elif self.state == "LISTENING":
            if "gym" in perception['speech_text']:
                self.state = "GUIDING"
                print("Robot: The gym is this way. Please follow me.")
                
        elif self.state == "GUIDING":
            # Check Gesture "Stop"
            if perception['hand_gesture'] == "OPEN_PALM":
                print("Robot: Pausing.")
                self.state = "PAUSED"
```

### 👨‍💻 Component 3: Safety Override (`src/safety_layer.py`)

Even if Brain says "Guide", Safety says "Stop".

```python
class SafetyLayer:
    def check(self, cmd_vel):
        scan = get_lidar()
        if min(scan) < 0.5:
            return 0.0, 0.0 # Override
        return cmd_vel # Pass through
```

### 👨‍💻 Main Loop (`run_receptionist.py`)

```python
brain = ReceptionistBrain()
perception = PerceptionEngine()
safety = SafetyLayer()

while True:
    # 1. Perceive
    user_state = perception.get_user_state()
    
    # 2. Think
    nav_cmd = brain.update(user_state)
    
    # 3. Safety Check
    safe_cmd = safety.check(nav_cmd)
    
    # 4. Act
    robot.drive(safe_cmd)
```

---

## 📝 Self-Assessment Quiz

1.  **Latency:**
    *   Whisper takes 2 seconds. The user walks away. What to do?
        *   **A:** Use VAD to detect end-of-speech immediately. Use "Streaming" ASR. Show visual feedback ("Listening...") so user knows to wait.

2.  **Privacy:**
    *   Where do we process the face images?
        *   **A:** On-Edge (Jetson). Do not send images to Cloud unless necessary. ISO 27001 compliance.

3.  **Safety:**
    *   User asks "Run into that wall".
    *   **A:** Brain says "Ok". Safety Layer says "No". Safety Layer must always be lower-level and privileged.

---

## ⏭️ Look Ahead: Week 8
We have a robot that can See, Plan, Talk, and Drive.
It's time to **Touch**.
**Week 8: Advanced Manipulation.**
*   6-DOF Arms.
*   Inverse Kinematics (MoveIt).
*   Gasping (GraspNet).
*   Force Control.

---

**Week 7 Complete**
