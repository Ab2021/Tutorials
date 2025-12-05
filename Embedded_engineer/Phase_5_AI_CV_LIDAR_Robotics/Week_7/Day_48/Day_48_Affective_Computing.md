# Day 48: Affective Computing (Emotion)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> Robots are cold. Humans are emotional. To bridge the gap, the robot must *feign* empathy.
> - **Focus:** Facial Expression Recognition (FER), Sentiment Analysis (VADER/BERT), and Emotion Synthesis.
> - **Code:** A system that detects User Emotion and adjusts Robot Personality (Happy/Sad/Neutral).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a FER model (DeepFace/Fer2013) to classify 7 basic emotions.
2.  **Integrate** VADER Sentiment Analysis for text/voice input.
3.  **Model** an Artificial Emotion System (PAD: Pleasure-Arousal-Dominance).
4.  **Create** Interactive Behaviors: If User = Sad, Robot = Slow & Sympathetic.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Webcam.

### Software Environment
```bash
pip install deepface nltk textblob
# Optional: Cozmo/Vector SDK (Animation)
```

### Prior Knowledge
- CNNs (Classification).
- NLP (Tokenization).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Emotion Recognition (Input)

1.  **Facial Expression (FER):**
    *   Ekman's 7 Universal Emotions: Happiness, Sadness, Anger, Fear, Disgust, Surprise, Neutral.
    *   **Architecture:** Crop Face $\to$ VGG/ResNet $\to$ Softmax(7).
2.  **Voice Prosody:**
    *   Pitch, volume, speed. (High pitch + Fast = Excitement/Fear).
3.  **Semantic Sentiment:**
    *   "I hate this!" $\to$ Negative.

### 🔹 Part 2: Emotion Synthesis (Output)

1.  **Categorical:** Robot State = HAPPY.
2.  **Dimensional (PAD Model):**
    *   **Pleasure:** Positive vs Negative.
    *   **Arousal:** Excited vs Calm.
    *   **Dominance:** Controlled vs Controlling.
    *   *Example:* Angry = Low P, High A, High D. Fear = Low P, High A, Low D.

### 🔹 Part 3: The Uncanny Valley

Mori (1970). As a robot looks more human, affinity rises. Until it looks *almost* human but slightly off (Zombie-like). Then affinity crashes.
*   **Rule:** If you can't be perfect, be cartoonish (Wall-E).

---

## 💻 Implementation: The Empathy Module

We will combine Face and Text analysis.

### 🛠️ Project Structure
```text
day48_emotion/
├── src/
│   ├── emotion_detector.py
│   ├── behavior_engine.py
│   └── text_analysis.py
└── run_empathy.py
```

### 👨‍💻 Facial Emotion (`src/emotion_detector.py`)

Using `DeepFace` library (Wraps Keras models).

```python
from deepface import DeepFace
import cv2

class FaceEmotion:
    def detect(self, img):
        try:
            # Enforce detection=False prevents crash if no face found
            objs = DeepFace.analyze(img, actions=['emotion'], 
                                  enforce_detection=False, silent=True)
            if not objs:
                return "neutral"
            
            # Return dominant emotion of first face
            return objs[0]['dominant_emotion']
        except:
            return "neutral"
```

### 👨‍💻 Text Sentiment (`src/text_analysis.py`)

Using NLTK VADER (Valence Aware Dictionary). Fast and ruled-based.

```python
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download lexicon once
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class TextEmotion:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze(self, text):
        scores = self.sia.polarity_scores(text)
        # Compound score: -1 (Negative) to +1 (Positive)
        return scores['compound']
```

### 👨‍💻 Behavior Code (`src/behavior_engine.py`)

```python
class RobotPersonality:
    def __init__(self):
        self.state = "NEUTRAL"
        self.led_color = (255, 255, 255) # White
        
    def update(self, face_emo, text_score):
        # 1. Fuse Inputs
        # Text overrides face? Combining Logic
        
        final_mood = "NEUTRAL"
        
        if face_emo == 'happy' or text_score > 0.5:
            final_mood = "HAPPY"
            self.led_color = (0, 255, 0) # Green
            
        elif face_emo in ['sad', 'fear'] or text_score < -0.5:
            final_mood = "SAD"
            self.led_color = (0, 0, 255) # Blue
            
        elif face_emo == 'angry':
            final_mood = "CAUTIOUS"
            self.led_color = (255, 0, 0) # Red
            
        self.state = final_mood
        print(f"Robot Mood: {self.state} (Face:{face_emo}, Text:{text_score:.2f})")
```

### 👨‍💻 Main Loop (`run_empathy.py`)

```python
import cv2
from src.emotion_detector import FaceEmotion
from src.text_analysis import TextEmotion
from src.behavior_engine import RobotPersonality

# Simulating Text Input for now (could connect to Whisper Day 43)
fake_speech = [
    "I am having a great day!", 
    "I just crashed my car.",
    "This is okay I guess."
]

face = FaceEmotion()
text = TextEmotion()
robot = RobotPersonality()
cap = cv2.VideoCapture(0)

i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Detect Face
    emo = face.detect(frame)
    if emo == "neutral": # Optimization
        # Only check deepface every 10 frames usually
         pass

    # 2. Fake Text input (Cycle through)
    txt_input = fake_speech[i % 3]
    score = text.analyze(txt_input)
    
    # 3. Update Robot
    robot.update(emo, score)
    
    # Draw
    cv2.rectangle(frame, (0,0), (640, 50), robot.led_color, -1)
    cv2.putText(frame, f"User: {emo.upper()}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    cv2.imshow("Empathy Bot", frame)
    if cv2.waitKey(100) == ord('q'): break
    
    if cv2.waitKey(1) == ord('n'): # Next text
        i += 1

cap.release()
cv2.destroyAllWindows()
```

---

## 🔬 Lab Exercise: The Mirroring Game

### 1. Lab Objectives
- Robot has a "Face" (Screen).
- **Task:** Mimic the user.
    - User Smiles $\to$ Robot displays `happy.png`.
    - User Frowns $\to$ Robot displays `sad.png`.
- **Psychology:** This builds rapport ("Mirroring").
- **Latency Challenge:** If mimicry > 500ms, it feels creepy/mocking. Must be fast.

---

## 🚀 Project: "Customer Service Robot"

**Goal:** Detect frustrated users.
1.  **Input:** Camera + Microphone.
2.  **Trigger:**
    - Face = Angry.
    - Voice = Loud (Volume > threshold).
    - Keywords = "Stupid", "Broken".
3.  **Action:** Escalation.
    - Robot: "I sense you are upset. Connecting you to a human operator."
    - Light turns Orange.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Resting Bitch Face" (RBF)
*   **Symptom:** Neutral faces detected as "Sad" or "Angry".
*   **Cause:** Bias in training data (Fer2013). Low resolution.
*   **Fix:** Calibrate to baseline. Calculate $\Delta \text{Emotion}$. "User is *more* sad than 10 seconds ago."

#### 2. "Lighting"
*   **Symptom:** Shadows under eyes look like Sadness features.
*   **Fix:** Histogram Equalization. Use Infrared (FaceID) cameras.

---

## ⚡ Optimization: Multi-Task Learning

Instead of running Face Detection, then Landmark, then Emotion separately.
*   Use **FaceMesh** (MediaPipe).
*   Extract distances (Lip corners).
*   Use a simple MLP classifier on landmarks.
*   Speed: 200 FPS vs 10 FPS (CNN).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the PAD model?
    *   **A:** Pleasure-Arousal-Dominance. A continuous 3D space to map emotions, unlike discrete categories.
2.  **Q:** Can a robot "feel" emotion?
    *   **A:** No. It simulates expressions to trigger emotional response in humans. This is "Affective Computing".
3.  **Q:** Why is Voice Prosody useful?
    *   **A:** Sarcasm detection. "Yeah, great job" (Low pitch, slow) $\to$ Negative sentiment, despite "Great" keyword.

### Challenge Task
> **Task:** Blink Detection (Liveness).
> 1. Compute Eye Aspect Ratio (EAR).
> 2. Detect sequence: Eye Open $\to$ Closed $\to$ Open.
> 3. Used to distinguish a Real Person vs a Photograph (Anti-Spoofing).

---

## 📚 Further Reading
- **Affective Computing:** Picard (1997).
- **The Uncanny Valley:** Mori (1970).

---

**Day 48 Complete**
