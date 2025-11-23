# Day 94: Creative Writing & Content Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a "Style Mimic" Agent

We will build an agent that analyzes a sample text and then writes about a new topic in that style.

```python
class StyleAgent:
    def __init__(self, client):
        self.client = client

    def analyze_style(self, sample_text):
        prompt = f"""
        Analyze the writing style of the following text. 
        Focus on: Sentence length, Vocabulary complexity, Tone (Formal/Casual), and Rhetorical devices.
        Output a concise "Style Guide".
        
        Text:
        {sample_text}
        """
        return self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

    def write(self, topic, style_guide):
        prompt = f"""
        Write a blog post about: {topic}
        
        Follow this Style Guide strictly:
        {style_guide}
        """
        return self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

# Usage
sample = "I'm sick of complex tech. Just give me a button. That's it."
agent = StyleAgent(client)
guide = agent.analyze_style(sample)
print(f"Style Guide: {guide}")
# Output: "Short, punchy sentences. Frustrated tone. Simple vocabulary."

post = agent.write("AI Agents", guide)
print(post)
# Output: "AI Agents are too hard. I don't want to configure them. Just work."
```

### The "Editor" Chain

Good writing comes from rewriting.
1.  **Drafter:** Writes the content.
2.  **Critic:** Checks against rules ("Did you use passive voice?").
3.  **Editor:** Rewrites based on critique.

```python
def write_polished_article(topic):
    draft = drafter.generate(topic)
    critique = critic.evaluate(draft)
    final = editor.rewrite(draft, critique)
    return final
```

### Constrained Generation (Grammars)

For SEO, you might need specific JSON-LD structures or strict HTML.
Using **Grammar-Constrained Decoding** (like `guidance` or `outlines`) ensures the output matches the schema perfectly.

### Summary

*   **Analysis -> Synthesis:** Don't just ask "Write like Shakespeare". Ask "Analyze Shakespeare, then write using that analysis."
*   **Iterative Refinement:** The best quality comes from a Draft-Critique-Edit loop.
