# Day 90: Foundation Models for Humanoids
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> The "ChatGPT moment" for Robotics.
> - **Focus:** Vision-Language-Action (VLA) Models, RT-2, NVIDIA GR00T, and Zero-Shot Generalization.
> - **Code:** A demo script using a VLM (like LLaVA or a distilled RT-2 stub) to parse natural language instructions into high-level robot primitives.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the architecture of a VLA (Transformer processing generic tokens for Text, Image, and Robot Action).
2.  **Use** a Large Language Model (LLM) to decompose complex tasks ("Clean the room") into primitives.
3.  **Simulate** a "Semantic Planner" which uses vision to verify if a sub-task is complete.
4.  **Discuss** the data requirements for General Purpose Robots (The "Open X-Embodiment" Dataset).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (>12GB VRAM) for running local VLMs.

### Software Environment
```bash
pip install transformers accelerate bitsandbytes
```

### Prior Knowledge
- Transformers (Day 43).
- Behavior Trees (Day 75).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: From Specialist to Generalist

*   **Specialist:** Trained on 10k hours of *walking*. Can't cook.
*   **Generalist (Foundation Model):** Trained on Internet Scale data (Youtube, Wikipedia) + Robot Data.
*   **Key Insight:** Reasoning transfers. If the model knows "Apples are Fragile" from Wikipedia, it knows "Don't crush the apple" in robotics, even if it never saw a robot crush an apple.

### 🔹 Part 2: Robotic Transformer 2 (RT-2)

Google DeepMind's approach.
*   **Input:** Image + Text "Put the strawberry in the correct bowl".
*   **Output:** Text Tokens representing Action Digits `[121, 005, 255...]`.
    *   These tokens map to discretized End-Effector $\Delta Pose$.
*   **Training:** Fine-tune a standard VLM (PaLI-X) on robot trajectories.

### 🔹 Part 3: NVIDIA GR00T

A Foundation Model specifically for Humanoids.
*   **Understanding:** Multi-modal instructions.
*   **Generation:** Low-level joint limits and balance awareness integrated.
*   **Simulation:** Trained in "Isaac Lab" with massive Domain Randomization.

---

## 💻 Implementation: LLM Planner

We will use a VLM (llava-hf/llava-1.5-7b-hf) to act as the "Cerebrum".
It won't output motor torques (too slow). It will output **Function Calls** for our Behavior Tree.

### 🛠️ Project Structure
```text
day90_foundation/
├── src/
│   ├── vla_brain.py
│   └── robot_skills.py
└── prompt/
    └── system_prompt.txt
```

### 👨‍💻 VLA Brain (`src/vla_brain.py`)

Takes an image and instruction. Outputs a JSON plan.

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

class VLABrain:
    def __init__(self):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        # Load in 4bit to save memory
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )

    def plan(self, image_path, instruction):
        image = Image.open(image_path)
        
        # System Prompt defines the "API" the robot supports
        prompt = """USER: <image>
You are a robot planner. Available skills:
- pick(object_name)
- place(object_name, location)
- open(object_name)
- navigate_to(location)

User Instruction: {instr}
Output a valid JSON list of steps.
ASSISTANT:"""
        
        prompt = prompt.replace("{instr}", instruction)
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        generate_ids = self.model.generate(
            **inputs, 
            max_new_tokens=200,
            do_sample=False
        )
        
        result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Parse JSON from result (simplified)
        return result.split("ASSISTANT:")[-1].strip()

# Usage
if __name__ == "__main__":
    brain = VLABrain()
    plan = brain.plan("scene_messy_kitchen.jpg", "Put the apple in the fridge")
    print(f"Generated Plan: {plan}")
    # Expected: 
    # [
    #   {"skill": "navigate_to", "arg": "apple"}, 
    #   {"skill": "pick", "arg": "apple"},
    #   {"skill": "navigate_to", "arg": "fridge"},
    #   {"skill": "open", "arg": "fridge"},
    #   {"skill": "place", "arg": "apple", "loc": "inside_fridge"}
    # ]
```

### 👨‍💻 Robot Skills (`src/robot_skills.py`)

The "Cerebellum". Execute the primitives.

```python
import time

def pick(obj):
    print(f"Skill: Picking up {obj} using GraspNet...")
    time.sleep(2)
    return True

def navigate_to(loc):
    print(f"Skill: Nav2 to {loc}...")
    time.sleep(3)
    return True
    
# ... The implementation connects to our Week 10/11 code.
```

---

## 🔬 Lab Exercise: "The Hallucination"

### 1. Lab Objectives
- **Task:** Ask VLA to "Fly to the moon".
- **Observation:** Model might output `{"skill": "fly_to", "arg": "moon"}`.
- **Fail:** Our API doesn't have `fly_to`.
- **Constraint:** Modify System Prompt: "If task is impossible, return IMPOSSIBLE".
- **Test:** Ask again.
- **Result:** Models are suggestible. Prompt Engineering is critical for safety.

---

## 🚀 Project: "Language Guided Patrol"

**Goal:** Natural Language Nav2.
1.  **Map:** Label coordinates: "Kitchen", "Lobby", "Lab".
2.  **Input:** Voice Command "Check if the Lab is empty".
3.  **VLA:** 
    *   Step 1: `navigate_to("Lab")`.
    *   Step 2: Take Photo.
    *   Step 3: VQA (Visual Question Answering) on photo: "Is there a person?"
    *   Step 4: Report back.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Model Output Parsing Error"
*   **Symptom:** LLM chats instead of outputting JSON. "Sure, I can help you with that! Here is the plan..."
*   **Fix:** Use "Grammar Constrained Decoding" (e.g., Guidance or Outlines library) to force the output to adhere to a JSON schema.

#### 2. "Latency"
*   **Symptom:** Robot pauses for 5 seconds between steps.
*   **Cause:** Inference on 7B model is slow.
*   **Fix:** Use quantized models (AWQ) or TensorRT-LLM. Run VLA asynchronously (Pipeline: Plan next step while acting current step).

---

## ⚡ Optimization: 3D-LLMs

2D Images lose depth.
*   **3D-LLM:** Takes Point Clouds as input.
*   **Understanding:** "Pick up the mug *behind* the laptop".
*   Essential for occlusion handling.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Zero-Shot"?
    *   **A:** Doing a task without ever being trained on that specific task. (e.g., Handling a brand new object).
2.  **Q:** Why not use ChatGPT (API)?
    *   **A:** Latency and Privacy. Also, no direct image input (unless GPT-4V). Local Multi-modal models are preferred for robotics.
3.  **Q:** What is the "Action Tokenization" in RT-2?
    *   **A:** Converting continuous values (0.54 rad) into discrete text tokens ("Token_54") so the LLM can predict them like words.

### Challenge Task
> **Task:** Visual Feedback Loop.
> 1. Plan: `pick("apple")`.
> 2. Execute.
> 3. Take Photo.
> 4. Ask VLA: "Did I pick up the apple?"
> 5. If No -> Retry.

---

## 📚 Further Reading
- **RT-2 Paper:** "Vision-Language-Action Models with Web-Scale Knowledge".
- **Voyager (Minecraft):** Lifelong learning agent using LLMs.

---

**Day 90 Complete**
