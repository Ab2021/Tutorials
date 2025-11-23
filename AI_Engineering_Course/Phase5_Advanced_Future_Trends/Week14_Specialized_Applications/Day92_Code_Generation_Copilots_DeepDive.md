# Day 92: Code Generation & Copilots
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a FIM Completion Engine

We will simulate how a Copilot constructs a prompt for a FIM model (like StarCoder).

```python
class CopilotEngine:
    def __init__(self, client, model="bigcode/starcoder"):
        self.client = client
        self.model = model
        self.FIM_PREFIX = "<fim_prefix>"
        self.FIM_SUFFIX = "<fim_suffix>"
        self.FIM_MIDDLE = "<fim_middle>"

    def construct_prompt(self, file_content, cursor_pos):
        # 1. Split content
        prefix = file_content[:cursor_pos]
        suffix = file_content[cursor_pos:]
        
        # 2. Construct FIM Prompt
        # Note: Different models use different tokens (PSM vs SPM)
        prompt = f"{self.FIM_PREFIX}{prefix}{self.FIM_SUFFIX}{suffix}{self.FIM_MIDDLE}"
        return prompt

    def complete(self, prompt):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=50,
            temperature=0.2,
            stop=[self.FIM_PREFIX, "\n\n"] # Stop at block end
        )
        return response.choices[0].text

# Usage
code = """
def calculate_area(radius):
    # Calculate area of circle
    
    return area
"""
cursor = code.find("    return area") # Cursor is before return
engine = CopilotEngine(client)
prompt = engine.construct_prompt(code, cursor)
print(f"Prompt: {prompt}")
# Output: <fim_prefix>...# Calculate area of circle\n    <fim_suffix>\n    return area<fim_middle>
```

### Building a "Repo Map" (Tree-Sitter)

To give the model context about other files, we use **Tree-Sitter** to parse the AST.

```python
from tree_sitter import Language, Parser

# Load Python Grammar
PY_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def get_definitions(code):
    tree = parser.parse(bytes(code, "utf8"))
    query = PY_LANGUAGE.query("""
    (function_definition
      name: (identifier) @func.name)
    (class_definition
      name: (identifier) @class.name)
    """)
    
    captures = query.captures(tree.root_node)
    return [node.text.decode('utf8') for node, _ in captures]

# This allows us to extract "signatures" from 100 files and pack them into the prompt.
```

### Speculative Decoding for Latency

Copilots need <50ms latency.
**Speculative Decoding:**
1.  **Draft Model:** A tiny model (Codex-10M) generates 5 tokens instantly.
2.  **Verify Model:** The big model (Codex-12B) verifies them in parallel.
3.  If correct, accept all 5. If not, rollback to the first error.
This doubles the effective tokens/sec.

### Summary

*   **FIM** is the standard format.
*   **Tree-Sitter** is the standard parser.
*   **Latency** is the primary metric. Users tolerate bad suggestions, but they don't tolerate lag.
