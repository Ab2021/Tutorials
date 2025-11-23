# Day 62: Testing & Debugging Multi-Agent Systems
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building a Multi-Agent Debugger

We will implement a simple **Trace Logger** and a **Unit Test** for a Handoff.

### 1. The Trace Logger

We need to capture every message.

```python
import json
import time

class Tracer:
    def __init__(self):
        self.events = []
        
    def log(self, source, target, message, type="message"):
        event = {
            "timestamp": time.time(),
            "source": source,
            "target": target,
            "message": message,
            "type": type
        }
        self.events.append(event)
        print(f"[{source} -> {target}]: {message[:50]}...")

    def save(self, filename="trace.json"):
        with open(filename, "w") as f:
            json.dump(self.events, f, indent=2)

tracer = Tracer()

# Usage in Agent
# tracer.log(self.name, recipient.name, message_content)
```

### 2. Unit Testing a Handoff (Mocking)

We want to test if the Triage Agent correctly hands off "Refund" requests to the Refund Agent.

```python
import unittest
from unittest.mock import MagicMock

class TestTriageAgent(unittest.TestCase):
    def test_handoff_logic(self):
        # 1. Setup
        triage = Agent("Triage", "Route refunds to RefundAgent.")
        mock_refund_agent = MagicMock()
        mock_refund_agent.name = "RefundAgent"
        
        # 2. Execute
        user_input = "I want my money back."
        response = triage.process(user_input)
        
        # 3. Assert
        # Check if the output contains the Handoff Signal
        self.assertIn("TRANSFER: RefundAgent", response)
        
        # Or if using a framework, check if the next speaker was set
        # self.assertEqual(groupchat.next_speaker, mock_refund_agent)

if __name__ == '__main__':
    unittest.main()
```

### 3. Automated Evaluation of Traces (LLM Judge)

After running a simulation, we use GPT-4 to analyze the `trace.json`.

```python
def analyze_trace(trace_file):
    with open(trace_file) as f:
        trace = json.load(f)
        
    trace_str = json.dumps(trace)
    
    prompt = f"""
    Analyze this conversation trace for bugs.
    Look for:
    1. Infinite loops.
    2. Agents forgetting instructions.
    3. Rude behavior.
    
    Trace:
    {trace_str}
    
    Report:
    """
    
    report = llm.invoke(prompt).content
    return report
```

### 4. Visualizing the Graph (LangGraph)

LangGraph has built-in visualization.

```python
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod

# Generate Mermaid Diagram
print(app.get_graph().draw_mermaid())

# Copy-paste this into the Mermaid Live Editor to see the flow.
# Useful for debugging "Why did it go to Node C instead of B?"
```

### Summary

*   **Tracer:** The flight recorder.
*   **Mocks:** Isolate the agent under test.
*   **LLM Judge:** Automated QA.
*   **Visualization:** See the map.
Without these, you are flying blind.
