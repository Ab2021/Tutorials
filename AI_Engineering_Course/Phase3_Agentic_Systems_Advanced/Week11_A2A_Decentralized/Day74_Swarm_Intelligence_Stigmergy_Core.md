# Day 74: Swarm Intelligence & Stigmergy
## Core Concepts & Theory

### What is a Swarm?

A Swarm is a system where complex global behavior emerges from simple local rules.
*   **Ants:** No "General Ant" tells the others what to do.
*   **Birds:** No "Lead Bird" coordinates the flock.
*   **Agents:** We can build AI systems that coordinate without a central Orchestrator.

### 1. Stigmergy (Indirect Coordination)

The key mechanism of swarms.
**Definition:** Coordination through the environment.
*   *Ant Example:* Ant A finds food. It drops a pheromone trail on the ground. Ant B smells the trail and follows it. Ant A and B never "talked".
*   *AI Example:* Agent A writes a partial solution to a shared Database. Agent B sees the entry and improves it.

### 2. Ant Colony Optimization (ACO)

A probabilistic technique for solving pathfinding problems.
*   Agents explore random paths.
*   Successful agents leave a "digital pheromone" (weight) on the path.
*   Future agents prefer paths with higher weights.
*   Over time, the swarm converges on the optimal path.

### 3. Boids (Flocking)

Simulating organic movement.
*   **Separation:** Steer to avoid crowding local flockmates.
*   **Alignment:** Steer towards the average heading of local flockmates.
*   **Cohesion:** Steer to move towards the average position of local flockmates.
Applied to AI: "Keep your output similar to your peers (Alignment) but don't duplicate work (Separation)."

### 4. Benefits of Swarms

*   **Robustness:** If 50% of agents die, the swarm continues.
*   **Scalability:** Adding more agents improves performance linearly (up to a point).
*   **Simplicity:** Individual agents can be very dumb (cheap models).

### Summary

Swarm Intelligence moves us from "Smart Agents" to "Smart Systems". It is ideal for problems like routing, search, and coverage where the environment is dynamic and central control is brittle.
