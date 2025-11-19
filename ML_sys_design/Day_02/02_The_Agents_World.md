# Day 2: The Agent's World - The PEAS Framework

On Day 1, we introduced the concept of an AI Agent. Today, we get formal. To properly design and evaluate an agent, we need a precise way to describe its task and its environment. The **PEAS framework** is the standard for doing this.

PEAS stands for:
*   **P**erformance Measure
*   **E**nvironment
*   **A**ctuators
*   **S**ensors

Mastering this framework is non-negotiable for an agentic AI engineer. It's the first thing you should reach for when presented with a new agent design problem. It forces you to clarify the requirements before writing a single line of code.

---

## Part 1: P - Performance Measure

The Performance Measure is the agent's "scorecard." It's a function that evaluates a sequence of states in the environment and tells us how well the agent is doing. It is the most important part of the PEAS framework, as it defines what we want the agent to achieve.

> **Key Idea:** The performance measure is about the *outcome*, not the agent's internal thoughts. A good performance measure is objective, measurable, and truly reflects the desired goal.

**Common Pitfalls in Defining Performance:**
*   **Being too narrow:** For a self-driving car, "stay in the lane" is a poor performance measure. It doesn't account for reaching the destination, passenger comfort, or safety.
*   **Including the agent's actions:** "Make lots of trades" is a bad measure for a stock-trading bot. The goal is to *make money*, which is an outcome, not an action.

**Example: A Vacuum Cleaner Agent**
*   *Poor Performance Measure:* "Move around the room a lot."
*   *Good Performance Measure:* Amount of dirt collected, percentage of floor space cleaned, electricity consumed, time taken.
*   *Excellent Performance Measure:* A combination of the above, weighted by importance (e.g., maximize dirt collected while minimizing time and power consumption).

---

## Part 2: E - Environment

The environment is the world the agent lives and operates in. The nature of the environment profoundly dictates the complexity of the agent. When analyzing an environment, we use several key properties:

### **1. Observable vs. Partially Observable**
*   **Fully Observable:** The agent's sensors can detect all aspects of the state of the environment that are relevant to its decision-making at any given point in time. (e.g., A chess game).
*   **Partially Observable:** The agent has incomplete information. (e.g., A self-driving car cannot see around corners; a poker-playing agent cannot see other players' hands). Most real-world environments are partially observable.

### **2. Deterministic vs. Stochastic**
*   **Deterministic:** The next state of the environment is completely determined by the current state and the agent's action. (e.g., A computer puzzle like Sudoku).
*   **Stochastic:** There is an element of randomness. The next state is not perfectly predictable. (e.g., A self-driving car—you can't predict exactly what other drivers will do). Most real-world environments are stochastic.

### **3. Episodic vs. Sequential**
*   **Episodic:** The agent's experience is divided into atomic "episodes." The agent's action in one episode does not affect the next one. (e.g., An image classification agent identifying objects in a series of unrelated photos).
*   **Sequential:** The current decision can affect all future decisions. The agent needs to think ahead. (e.g., Chess, a self-driving car, a customer service conversation). Most interesting agentic tasks are sequential.

### **4. Static vs. Dynamic**
*   **Static:** The environment does not change while the agent is "thinking." (e.g., A crossword puzzle).
*   **Dynamic:** The environment can change while the agent is deciding on an action. The agent must be able to react in real-time. (e.g., A self-driving car in traffic).

### **5. Discrete vs. Continuous**
*   **Discrete:** The number of possible percepts and actions is finite. (e.g., Chess, a crossword puzzle).
*   **Continuous:** The state and actions can range over continuous values. (e.g., The steering, acceleration, and braking of a self-driving car).

---

## Part 3: A - Actuators

Actuators are the mechanisms by which the agent **acts upon** its environment. They are the agent's "hands" or "muscles."

*   **Software Agents:** The actuators are API calls, shell commands, writing to a file, sending an email, or displaying something on a screen.
*   **Robotic Agents:** The actuators are motors, grippers, wheels, and speakers.

When defining actuators, you must consider their **range, precision, and potential for failure.** An API call can fail; a robotic arm has a limited range of motion.

---

## Part 4: S - Sensors

Sensors are the mechanisms by which the agent **perceives** its environment. They are the agent's "senses."

*   **Software Agents:** The sensors are reading a file, receiving an API response, parsing the HTML of a webpage, or getting user input from a text box.
*   **Robotic Agents:** The sensors are cameras, microphones, accelerometers, GPS, and touch sensors.

When defining sensors, you must consider their **fidelity and potential for noise.** A camera has limited resolution and can be affected by lighting; an API response might be incomplete or malformed.

---

## Activity: PEAS Framework for Real-World Agents

Let's put it all together. For each of the agents below, create a detailed PEAS description. Be specific.

### **Agent 1: A Stock Trading Agent**
*   **Performance Measure:**
*   **Environment:**
*   **Actuators:**
*   **Sensors:**

### **Agent 2: An Email Spam Filter Agent**
*   **Performance Measure:**
*   **Environment:**
*   **Actuators:**
*   **Sensors:**

### **Agent 3: A Medical Diagnosis Agent**
*   **Performance Measure:**
*   **Environment:**
*   **Actuators:**
*   **Sensors:**

**Example Answer for Agent 1: Stock Trading Agent**
*   **Performance Measure:** Maximize portfolio value over a 1-year period; minimize risk (volatility) below a certain threshold.
*   **Environment:** The stock market. It is **partially observable** (we don't know other traders' secret strategies), **stochastic** (prices fluctuate unpredictably), **sequential** (today's trades affect tomorrow's capital), **dynamic** (prices change while the agent is thinking), and **continuous** (prices and trade volumes are continuous variables).
*   **Actuators:** API calls to a brokerage to execute buy/sell orders (e.g., `POST /orders {symbol: "GOOG", quantity: 100, side: "buy"}`).
*   **Sensors:** API endpoints that provide real-time price feeds, historical price data, and news sentiment analysis scores.
