# Day 3, Topic 2: An Expert's Guide to the Reflection/Critique Pattern

## 1. The Philosophy of Reflection: Learning from Experience

Reflection is a cornerstone of human learning. As the educational theorist Donald Schön argued in his work on "reflective practice," we learn not just by doing, but by reflecting on what we have done. This process of "thinking about our thinking" allows us to identify our mistakes, to understand why we made them, and to develop new strategies for improving our performance in the future.

The Reflection/Critique pattern applies this same philosophy to AI agents. It is based on the idea that an agent can improve its performance by generating an output, critiquing that output, and then using the critique to generate a revised and improved output.

## 2. A Taxonomy of Reflection Strategies

*   **Self-Critique:** The simplest form of reflection, where an agent evaluates its own output.
*   **External Critique:** A separate "critic" agent evaluates the output of a "worker" agent. This can be more effective than self-critique, as it allows for a diversity of perspectives.
*   **Human-in-the-Loop Critique:** A human provides feedback on the agent's output. This is the most reliable form of critique, but it is also the most expensive.
*   **Tool-Interactive Critiquing:** An agent uses tools to validate its output. For example, a code-generating agent might run its generated code through a linter or a test suite to check for errors.

## 3. The "Reflexion" Framework: A Case Study

A key paper in this area is "Reflexion: Language Agents with Verbal Reinforcement Learning" by Shinn et al. The Reflexion framework enables an agent to learn from its mistakes through a process of "verbal reinforcement learning."

Instead of using a numerical reward signal, the Reflexion agent uses a text-based critique of its own performance to update its internal "memory" of what works and what doesn't. This allows the agent to learn and adapt its behavior without the need for expensive model fine-tuning.

## 4. Advanced Reflection Techniques

*   **Multi-turn Reflection:** An agent can engage in a multi-turn dialogue with itself to refine its thoughts. This is analogous to the human process of "thinking things through."
*   **Constitutional AI:** The idea of giving an agent a set of principles or a "constitution" to guide its self-correction process. For example, an agent might be given a constitution that instructs it to be helpful, harmless, and honest.
*   **The Role of Feedback:** The reflection process is driven by feedback. This can be human feedback, environmental feedback (e.g., the result of a tool call), or model-generated feedback (e.g., the output of a critic agent).

## 5. Real-World Applications of Reflective Agents

*   **Improving the accuracy of code generation:** A reflective agent can write a piece of code, run it through a test suite, and then use the test results to debug and improve the code.
*   **Enhancing the quality of creative writing:** A reflective agent can write a story, critique it for plot holes and inconsistencies, and then revise it to make it more compelling.
*   **Increasing the robustness of question-answering systems:** A reflective agent can answer a question, search for evidence to support its answer, and then revise its answer if it finds conflicting information.

## 6. Code Example

```python
# This is a conceptual example of a worker-critic interaction.

def reflective_agent(query, tools):
    # 1. Worker generates initial output
    worker_prompt = f"Question: {query}"
    initial_output = llm_worker(worker_prompt)

    # 2. Critic critiques the output
    critic_prompt = f"Please critique the following output: {initial_output}"
    critique = llm_critic(critic_prompt)

    # 3. Worker revises the output based on the critique
    reviser_prompt = f"Original query: {query}\nInitial output: {initial_output}\nCritique: {critique}\nPlease provide a revised and improved output."
    revised_output = llm_worker(reviser_prompt)

    return revised_output
```

## 7. Exercises

1.  Implement a simple two-agent critique system where one agent writes a short paragraph on a given topic, and a second agent critiques it.
2.  How could you use the "Constitutional AI" approach to build a safer and more ethical agent?

## 8. Further Reading and References

*   Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv preprint arXiv:2303.11366.
*   Madaan, A., et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv preprint arXiv:2303.17651.
*   Gou, Z., et al. (2024). *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*. arXiv preprint arXiv:2402.02339.