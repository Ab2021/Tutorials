# Day 63: Capstone: Building a Software Dev Swarm
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing the Swarm

We will build the **Software Dev Swarm** using a custom loop to maximize control.

### 1. The Shared State (FileSystem)

We simulate a file system using a dictionary for simplicity, but in prod, use real files.

```python
class VirtualFileSystem:
    def __init__(self):
        self.files = {}
        
    def write(self, path, content):
        self.files[path] = content
        return f"Wrote {len(content)} chars to {path}"
        
    def read(self, path):
        return self.files.get(path, "Error: File not found")
        
    def list_files(self):
        return list(self.files.keys())

fs = VirtualFileSystem()
```

### 2. The Agents

We define the 3 key roles.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Prompts
PM_PROMPT = "You are a Product Manager. Create a spec for: {goal}"
ARCHITECT_PROMPT = "You are an Architect. Given the spec: {spec}, list the files needed."
ENGINEER_PROMPT = """
You are an Engineer. Write the code for file: {filename}.
Context: {spec}
Existing Files: {file_list}
"""

def run_pm(goal):
    return llm.invoke(PM_PROMPT.format(goal=goal)).content

def run_architect(spec):
    return llm.invoke(ARCHITECT_PROMPT.format(spec=spec)).content

def run_engineer(filename, spec, file_list):
    return llm.invoke(ENGINEER_PROMPT.format(
        filename=filename, spec=spec, file_list=file_list
    )).content
```

### 3. The Orchestration Loop

```python
def run_swarm(goal):
    print(f"🚀 Starting Swarm for: {goal}")
    
    # Step 1: PM
    spec = run_pm(goal)
    fs.write("spec.md", spec)
    print("✅ Spec created.")
    
    # Step 2: Architect
    file_plan = run_architect(spec)
    # Parse the plan (Assume JSON list of filenames)
    # filenames = json.loads(file_plan) 
    filenames = ["game.py", "utils.py"] # Mocked for simplicity
    print(f"✅ Architecture planned: {filenames}")
    
    # Step 3: Engineer Loop (Sequential)
    for fname in filenames:
        print(f"👨‍💻 Writing {fname}...")
        # Engineer needs to see what has been written so far
        existing = fs.list_files()
        code = run_engineer(fname, spec, existing)
        fs.write(fname, code)
        
    print("🎉 Swarm Finished!")
    return fs.files

# Usage
# files = run_swarm("A Snake game in Python")
# print(files['game.py'])
```

### 4. Adding the QA Loop (Self-Correction)

The Engineer shouldn't just write; they should fix.

```python
def run_qa(filename, code):
    # Simulating a syntax check
    try:
        compile(code, filename, 'exec')
        return "PASS"
    except Exception as e:
        return f"FAIL: {str(e)}"

def robust_engineer_step(fname, spec):
    attempts = 0
    error = None
    
    while attempts < 3:
        # Generate
        if error:
            prompt = f"Fix this error in {fname}: {error}"
        else:
            prompt = f"Write code for {fname}"
            
        code = llm.invoke(prompt).content
        
        # Test
        result = run_qa(fname, code)
        if result == "PASS":
            return code
        
        error = result
        attempts += 1
        print(f"⚠️ Bug found in {fname}. Retrying...")
        
    raise Exception(f"Failed to write {fname} after 3 attempts.")
```

### 5. Advanced: Dependency Graph

If `game.py` imports `utils.py`, we must write `utils.py` **first**.
*   **Topological Sort:** The Architect should output a DAG (Directed Acyclic Graph) of dependencies.
*   **Execution Order:** The Orchestrator sorts the DAG and assigns tasks in that order.

### Summary

Building a Dev Swarm is about **Process Engineering**. The LLMs are just the workers; you are the Factory Designer. You must define the assembly line (Spec -> Arch -> Code -> QA) and the quality checks at each station.
