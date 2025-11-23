# Day 102: Robotics & Embodied AI
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a VLA Controller (Mock)

We will simulate how a VLA model translates text to actions.

```python
class RobotController:
    def __init__(self, client):
        self.client = client

    def act(self, image_desc, command):
        prompt = f"""
        You are a robot arm.
        State: {image_desc}
        Goal: {command}
        
        Available Actions: MOVE_X(cm), MOVE_Y(cm), GRIP_OPEN, GRIP_CLOSE.
        
        Output a sequence of actions.
        """
        return self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

# Usage
state = "Red block at (10, 10). Blue bin at (20, 20)."
cmd = "Put the red block in the blue bin."
robot = RobotController(client)
print(robot.act(state, cmd))
# Output:
# MOVE_X(10)
# MOVE_Y(10)
# GRIP_CLOSE
# MOVE_X(20)
# MOVE_Y(20)
# GRIP_OPEN
```

### Eureka: LLM as Reward Designer

In RL, designing the reward function is hard.
*   **Human:** `reward = dist(hand, object) + ...` (Hard to tune).
*   **Eureka:**
    1.  LLM writes Python code for a reward function.
    2.  RL agent trains using that reward.
    3.  LLM looks at the success rate.
    4.  LLM rewrites the code to improve it.
    *   *Result:* LLM finds "Pen Spinning" rewards better than humans.

### Visual Servoing

Using vision feedback loop.
1.  **Predict:** Where should the hand be?
2.  **Move:** Move slightly.
3.  **Observe:** Did the hand move correctly?
4.  **Correct:** Adjust error.

### Summary

*   **Tokenization:** The key breakthrough was tokenizing actions (discretizing continuous motor values into 256 bins) so Transformers can predict them.
