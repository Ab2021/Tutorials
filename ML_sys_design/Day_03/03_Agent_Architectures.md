# Day 3: Inside the Agent's Mind - A Tour of Agent Architectures

If the PEAS framework describes *what* an agent is supposed to do, the **agent architecture** describes *how* it does it. The architecture is the blueprint of the agent's internal structure, defining the relationship between its percepts, its internal state, its reasoning, and its actions.

Today we will explore the classical agent architectures. While modern agents are powered by LLMs, they are almost always a sophisticated implementation or hybrid of these fundamental designs.

---

## 1. Simple Reflex Agents

This is the simplest possible kind of agent. Its decisions are based *only* on the current percept. It has no memory of the past.

*   **Logic:** `IF condition THEN action`
*   **Structure:** Percept -> Condition-Action Rules -> Action
*   **Analogy:** A doctor's reflex hammer. The doctor taps your knee (percept), and your leg kicks (action). There is no thought or memory involved.

**Example: A Simple Email Filter**
*   **Percept:** An incoming email.
*   **Rule:** `IF the email body contains "you have won a lottery" THEN action is "move to spam folder".`
*   **Strength:** Very fast and simple to implement.
*   **Weakness:** Only works in fully observable environments. If our email filter can't see the sender's reputation (a past percept), it might wrongly classify a legitimate email from a friend joking about a lottery.

```python
def simple_reflex_email_agent(email_percept):
    """
    A simple reflex agent that decides based only on the current email.
    """
    if "you have won a lottery" in email_percept['body']:
        return "move_to_spam"
    else:
        return "move_to_inbox"

# Usage
email = {'from': 'spammer@scam.com', 'body': 'congratulations you have won a lottery'}
action = simple_reflex_email_agent(email)
print(f"Action: {action}") # -> "move_to_spam"
```

---

## 2. Model-Based Reflex Agents

This is a significant upgrade. A model-based agent maintains an **internal state** or **model** of the world. This model helps it handle partial observability by tracking things it can't currently see.

*   **Logic:** It updates its internal state based on the current percept and its past state, and then chooses an action.
*   **Structure:** Percept -> Update Internal State -> Condition-Action Rules -> Action
*   **Analogy:** Driving a car. You can't see the car in your blind spot right now, but you *remember* seeing it a moment ago, so your internal "model" of the road includes that car.

**How the model is updated:**
1.  **How the world evolves:** The model needs to know how the world changes on its own (e.g., "cars in front of me also move forward").
2.  **How my actions affect the world:** The model needs to predict the outcome of its own actions (e.g., "If I press the brake, my car will slow down").

**Example: A Smarter Thermostat**
*   **Percept:** The current temperature is 68°F.
*   **Internal State:** `{ 'current_temp': 68, 'last_seen_outside_temp': 30, 'is_heating_on': False, 'time_since_last_change': '5 minutes' }`
*   **Rule:** `IF current_temp < target_temp AND is_heating_on is False THEN turn on heater.`
*   **Strength:** Can make much better decisions in partially observable environments.

```python
class ModelBasedThermostatAgent:
    def __init__(self, target_temp=70):
        # Internal state (our model of the world)
        self.state = {
            'is_heating_on': False,
            'target_temp': target_temp
        }

    def update_state(self, percept):
        # Update our model based on new sensor information
        self.state['current_temp'] = percept['temperature']

    def choose_action(self, percept):
        self.update_state(percept)
        if self.state['current_temp'] < self.state['target_temp'] and not self.state['is_heating_on']:
            self.state['is_heating_on'] = True
            return "turn_on_heater"
        elif self.state['current_temp'] >= self.state['target_temp'] and self.state['is_heating_on']:
            self.state['is_heating_on'] = False
            return "turn_off_heater"
        else:
            return "do_nothing"

# Usage
agent = ModelBasedThermostatAgent()
percept1 = {'temperature': 68}
action1 = agent.choose_action(percept1)
print(f"Temp is 68. Action: {action1}") # -> "turn_on_heater"
print(f"Internal state: {agent.state}")

percept2 = {'temperature': 71}
action2 = agent.choose_action(percept2)
print(f"Temp is 71. Action: {action2}") # -> "turn_off_heater"
print(f"Internal state: {agent.state}")
```

---

## 3. Goal-Based Agents

While a model-based agent knows *how* the world works, a goal-based agent also knows *what it wants to achieve*. It combines its model with a **goal** to perform planning.

*   **Logic:** "Out of all the possible actions I can take, which one will get me closer to my goal?"
*   **Structure:** Percept -> Update Internal State -> **Plan a sequence of actions to achieve Goal** -> Execute first action
*   **Analogy:** Using a GPS. You tell it your destination (goal). It uses its map (model) to calculate a route (plan) and tells you the first step ("Turn left in 200 feet").
*   **Key Technology:** Search and Planning algorithms (e.g., A*, Dijkstra's) are fundamental to goal-based agents.

**Example: A Package Delivery Robot**
*   **Goal:** `{'location': 'Warehouse D'}`
*   **Model:** A map of the city, its current location, and the status of its battery.
*   **Plan:** It will simulate various paths (e.g., take Main Street, take the highway) and choose the sequence of turns that results in reaching Warehouse D in the shortest time. It then executes the first action of that plan.
*   **Strength:** Far more flexible than reflex agents. If a road is blocked, it can re-plan to find a new way to its goal.

---

## 4. Utility-Based Agents

Sometimes, just achieving the goal isn't enough. There can be many ways to reach a goal, but some are better than others. A utility-based agent aims for the *best* outcome.

*   **Logic:** It chooses the action that **maximizes its expected utility**. Utility is a function that maps a state to a real number, representing a degree of "happiness" or "desirability."
*   **Structure:** Similar to a goal-based agent, but instead of just a goal, it has a complex **utility function**.
*   **Analogy:** Using a GPS with settings. Do you want the *fastest* route, the *shortest* route, or the route with no tolls? Each of these is a different utility function. The fastest route might have tolls and be longer, but it's "better" according to your preference.

**Example: An Automated Taxi**
*   **Goal:** Get the passenger to their destination.
*   **Utility Function:** A function that seeks to maximize profit and passenger satisfaction while minimizing time, fuel consumption, and risk.
    *   `Utility = (fare_price) + (5-star_rating_probability * $5_bonus) - (fuel_cost) - (risk_of_accident_penalty)`
*   **Decision:** The agent might choose a slightly longer, safer route over a short but dangerous one because the safer route maximizes its expected utility, even if it doesn't achieve the "fastest time" goal.
*   **Strength:** Can make nuanced decisions in complex situations with trade-offs. This is the most sophisticated type of classical agent.

---

## 5. Learning Agents

A learning agent is not a separate architecture, but rather an enhancement to the other types. It is an agent that can improve its performance over time.

*   **Structure:** It has four main components:
    1.  **Performance Element:** The part of the agent that actually chooses actions (e.g., a model-based or utility-based component).
    2.  **Critic:** Provides feedback to the learning element. It compares the agent's performance against the performance measure and notes any discrepancy.
    3.  **Learning Element:** Responsible for making improvements to the performance element based on feedback from the critic.
    4.  **Problem Generator:** Responsible for suggesting new, exploratory actions to try, which can lead to new and better experiences.

**Example: A Chess-Playing Agent**
*   **Performance Element:** The current chess-playing strategy.
*   **Critic:** After a game, it observes whether the agent won or lost.
*   **Learning Element:** If the agent lost, it adjusts the weights in its strategy to make the moves that led to the loss less likely in the future.
*   **Problem Generator:** It might try a new, unconventional opening to see if it leads to a better outcome.

## Activity: Choose the Right Architecture

For each scenario below, which agent architecture would you choose as a starting point, and why?

1.  **An automated trading agent** that needs to balance high returns with low risk.
2.  **A smart vacuum cleaner** that needs to clean an entire room it has never seen before.
3.  **An agent that plays the game of Pac-Man.**
4.  **A system that automatically moderates a live chat** by deleting messages with profanity.
