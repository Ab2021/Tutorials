# Lab 02: GridWorld Environment

## Difficulty
🟢 Easy

## Problem Statement
Create a simple GridWorld environment class.
- Grid size: 4x4.
- Start: (0, 0), Goal: (3, 3).
- Actions: Up, Down, Left, Right.
- Reward: -1 per step, +10 at goal.
- `step(action)` returns `next_state`, `reward`, `done`.

## Starter Code
```python
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        
    def step(self, action):
        # TODO: Update state based on action
        # TODO: Check boundaries
        # TODO: Return (state, reward, done)
        pass
```
