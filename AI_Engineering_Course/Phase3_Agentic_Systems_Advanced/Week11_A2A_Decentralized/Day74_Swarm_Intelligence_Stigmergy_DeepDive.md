# Day 74: Swarm Intelligence & Stigmergy
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Stigmergy (Digital Pheromones)

We will simulate a "Search Swarm" looking for a target in a grid.
They communicate *only* by modifying the grid (Environment).

```python
import numpy as np
import random

GRID_SIZE = 20
PHEROMONE_DECAY = 0.95

class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE)) # Pheromone map
        self.target = (15, 15)
        
    def update(self):
        # Evaporation
        self.grid *= PHEROMONE_DECAY
        
    def add_pheromone(self, x, y, amount):
        self.grid[x, y] += amount

    def get_pheromone(self, x, y):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            return self.grid[x, y]
        return 0

class AntAgent:
    def __init__(self, env, x, y):
        self.env = env
        self.x = x
        self.y = y
        self.has_found_target = False

    def move(self):
        if self.has_found_target:
            # Drop pheromone on current spot
            self.env.add_pheromone(self.x, self.y, 1.0)
            # Move randomly (or back to base)
            self.random_move()
            return

        # Check if found target
        if (self.x, self.y) == self.env.target:
            self.has_found_target = True
            print("Target Found!")
            return

        # Smell neighbors
        best_move = None
        max_phero = -1
        
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        random.shuffle(moves)
        
        for dx, dy in moves:
            nx, ny = self.x + dx, self.y + dy
            p = self.env.get_pheromone(nx, ny)
            if p > max_phero:
                max_phero = p
                best_move = (nx, ny)
        
        # Probabilistic move: If pheromone is strong, follow it. Else, explore.
        if max_phero > 0.1 and random.random() < 0.8:
            self.x, self.y = best_move
        else:
            self.random_move()

    def random_move(self):
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.x = max(0, min(GRID_SIZE-1, self.x + dx))
        self.y = max(0, min(GRID_SIZE-1, self.y + dy))

# Simulation
env = Environment()
ants = [AntAgent(env, 0, 0) for _ in range(10)]

for step in range(100):
    env.update()
    for ant in ants:
        ant.move()
    # Visualization would go here
```

### Application: Code Refactoring Swarm

How does this apply to LLMs?
**The Codebase is the Environment.**
1.  **Linter Agent:** Wanders files. If it finds a messy function, it marks it with a `TODO: Refactor` comment (Pheromone).
2.  **Refactor Agent:** Wanders files. If it sees `TODO: Refactor`, it fixes the code and removes the comment.
3.  **Test Agent:** Wanders files. If a file was just changed, it runs tests. If fail, it adds `TODO: Fix Bug`.

This decouples the agents. The Linter doesn't need to know the Refactor Agent exists. It just modifies the environment.

### Summary

*   **Decoupling:** Producers and Consumers don't talk directly.
*   **Persistence:** The message (Pheromone) stays in the environment even if the sender dies.
*   **Emergence:** Complex workflows (Lint -> Refactor -> Test) emerge from simple rules.
