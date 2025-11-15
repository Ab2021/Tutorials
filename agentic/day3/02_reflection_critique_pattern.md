# Day 3, Topic 2: The Reflection/Critique Pattern

Agents, especially those based on LLMs, can sometimes produce outputs that are incorrect, incomplete, or suboptimal. The **Reflection/Critique Pattern** is a powerful technique for addressing this by enabling agents to iteratively improve their own work.

This pattern involves having an agent (or a separate "critic" agent) review and critique an output, and then use that feedback to generate a better version.

## Enabling Agents to Learn from Experience

The core idea behind the Reflection/Critique Pattern is to mimic the human process of self-improvement. When we work on a task, we often:

1.  Produce a first draft.
2.  Review and critique our own work.
3.  Identify areas for improvement.
4.  Revise the work based on our critique.

The Reflection/Critique Pattern applies this same process to AI agents. It allows an agent to "learn from its experience" within the context of a single task, leading to higher-quality outputs.

## How it Works: The Reflection Loop

The pattern typically works as follows:

1.  **Initial Output Generation:** The agent generates an initial output for a given task. This could be a piece of text, a plan, a piece of code, or any other output.
2.  **Reflection and Critique:** The agent is then prompted to reflect on and critique its own output. This is the key step. The prompt might ask questions like:
    *   "Review the previous response. Are there any errors or inconsistencies?"
    *   "Is this the best possible output, or are there ways to improve it?"
    *   "Does the output fully address the user's request?"
3.  **Feedback Generation:** The LLM, in "reflection mode," generates a critique of the initial output. This critique identifies specific flaws or areas for improvement.
4.  **Revision:** The agent is then given the original prompt, the initial output, and the critique, and is asked to generate a revised and improved output.

This loop can be repeated multiple times, with the agent iteratively refining its output until it reaches a satisfactory level of quality.

## A Concrete Example: Improving a Biography

Let's walk through a concrete example of the Reflection/Critique pattern in action, using a separate "critic" agent to improve a short biography of Marie Curie.

### 1. The Worker Prompt

```
Write a short biography of Marie Curie.
```

### 2. The Initial Output (from the Worker Agent)

```
Marie Curie was a famous scientist. She did important research on radioactivity. She was the first woman to win a Nobel Prize.
```

This output is factually correct, but it's a bit simplistic and lacks detail.

### 3. The Critic Prompt

Now, we'll use a "critic" agent to review this initial output. The critic is given a specific prompt that encourages it to be thorough and to provide actionable feedback.

```
You are a helpful assistant who reviews and critiques texts. Please review the following biography of Marie Curie and provide feedback on how it could be improved. Be specific and provide actionable suggestions.

Biography:
"Marie Curie was a famous scientist. She did important research on radioactivity. She was the first woman to win a Nobel Prize."
```

### 4. The Critique (from the Critic Agent)

```
The biography is a good start, but it could be improved in several ways:

1.  **Add more specific details:** The biography is very general. It could be improved by adding more specific details about her life and work. For example, what were her major discoveries? What were the names of the elements she discovered?
2.  **Mention her second Nobel Prize:** The biography only mentions that she was the first woman to win a Nobel Prize, but it doesn't mention that she won a second one in a different field, which is a very significant achievement.
3.  **Elaborate on her impact:** The biography mentions that she did "important research," but it doesn't explain what the impact of that research was. How did her work change the world?
4.  **Improve the flow and style:** The sentences are a bit choppy. The biography could be improved by using more varied sentence structures and transition words to make it flow more smoothly.
```

### 5. The Revision Prompt

Finally, we'll give the original prompt, the initial output, and the critique to a "reviser" agent (which could be the same as the worker agent) and ask it to generate a revised output.

```
Please revise the following biography of Marie Curie based on the provided critique.

Original Biography:
"Marie Curie was a famous scientist. She did important research on radioactivity. She was the first woman to win a Nobel Prize."

Critique:
1.  **Add more specific details:** The biography is very general. It could be improved by adding more specific details about her life and work. For example, what were her major discoveries? What were the names of the elements she discovered?
2.  **Mention her second Nobel Prize:** The biography only mentions that she was the first woman to win a Nobel Prize, but it doesn't mention that she won a second one in a different field, which is a very significant achievement.
3.  **Elaborate on her impact:** The biography mentions that she did "important research," but it doesn't explain what the impact of that research was. How did her work change the world?
4.  **Improve the flow and style:** The sentences are a bit choppy. The biography could be improved by using more varied sentence structures and transition words to make it flow more smoothly.
```

### 6. The Revised Output

```
Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity. Her work led to the discovery of two new elements, polonium and radium. A true trailblazer, she was the first woman to win a Nobel Prize, and she remains the only person to have won Nobel Prizes in two different scientific fields (Physics in 1903 and Chemistry in 1911). Her research not only revolutionized our understanding of the atom but also laid the foundation for modern nuclear physics and cancer treatments.
```

As you can see, the revised output is much more detailed, informative, and well-written than the initial output. This is the power of the Reflection/Critique pattern.

## A Word of Caution: The Risk of Over-Correction

While the Reflection/Critique pattern is very powerful, it's important to be aware of the risk of **over-correction**. This is a situation where an agent gets stuck in a loop of critiquing and re-writing its own work without making any substantial progress. The agent might make a series of small, insignificant changes, or it might even make the output worse in its attempt to "improve" it.

To mitigate this risk, it's a good idea to:

*   **Limit the number of reflection cycles:** Don't allow the agent to reflect on its own work indefinitely. Limit it to one or two cycles of reflection, and then take the best output.
*   **Use a separate "judge" agent:** You can use a third agent to act as a "judge" that compares the initial output with the revised output and decides which one is better.
*   **Incorporate human feedback:** Ultimately, the best way to avoid over-correction is to incorporate human feedback into the process. A human can provide the common-sense judgment needed to decide when an output is "good enough."



## Storing and Using Feedback

The feedback generated during the reflection process can also be stored and used to improve the agent's performance on future tasks. This is a form of online learning, where the agent continuously improves over time.

For example, if an agent consistently makes a particular type of error, and this error is repeatedly identified during the reflection process, the system could be designed to automatically fine-tune the agent's base model to correct for this bias.

## Exercise

1.  **Take a previously generated output from an agent (e.g., the travel plan from the morning session).**
2.  **Write a prompt to an LLM to act as a "critic" and provide feedback on the plan.**
    *   *Your prompt should encourage the critic to be thorough and to identify specific, actionable areas for improvement.*
3.  **Use the feedback to improve the original plan.**
    *   *You can do this manually, or you can write another prompt that gives the LLM the original plan and the critique, and asks it to generate a revised plan.*
