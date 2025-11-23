# Day 79: Specialized Domain Agents (Code, Legal, Medical)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Simple Coding Agent (File Editor)

Agent that can read and write files to solve a task.

```python
import os

class CodeAgentTools:
    def list_files(self, path="."):
        return str(os.listdir(path))
        
    def read_file(self, path):
        with open(path, "r") as f:
            return f.read()
            
    def write_file(self, path, content):
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote to {path}"

def coding_agent_loop(llm, task):
    tools = CodeAgentTools()
    history = [f"Task: {task}"]
    
    for i in range(5):
        prompt = "\n".join(history) + "\nNext Action (list, read, write):"
        action = llm.generate(prompt)
        
        if "list" in action:
            result = tools.list_files()
        elif "read" in action:
            # Parse path
            path = action.split(" ")[1]
            result = tools.read_file(path)
        elif "write" in action:
            # Parse path/content
            result = tools.write_file("test.py", "print('hello')")
        else:
            result = "Unknown action"
            
        history.append(f"Action: {action}\nResult: {result}")
        
        if "DONE" in action:
            break
            
    return history
```

### 2. Repo Map Generator (Context Compression)

Creating a tree view of the codebase for the LLM.

```python
import os

def generate_repo_map(root_dir, max_depth=2):
    tree = []
    
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        if level > max_depth:
            continue
            
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith(".py"):
                tree.append(f"{subindent}{f}")
                
    return "\n".join(tree)

# Usage
# print(generate_repo_map("./my_project"))
# Output:
# my_project/
#     src/
#         main.py
#         utils.py
```

### 3. Legal Citation Checker (Hallucination Guard)

Verifying citations against a database.

```python
class LegalVerifier:
    def __init__(self, case_db):
        self.case_db = case_db # List of valid case names
        
    def verify_citations(self, text):
        # Extract potential citations (Regex)
        import re
        citations = re.findall(r"[A-Z][a-zA-Z\s]+v\.\s[A-Z][a-zA-Z\s]+", text)
        
        invalid = []
        for cit in citations:
            if cit not in self.case_db:
                invalid.append(cit)
                
        if invalid:
            return False, f"Invalid Citations found: {invalid}"
        return True, "All citations verified."

# Usage
# verifier = LegalVerifier(["Roe v. Wade", "Brown v. Board"])
# valid, msg = verifier.verify_citations("According to Fake v. Case...")
```

### 4. Medical Note Summarizer (Scribe)

Structuring raw conversation into SOAP note.

```python
def generate_soap_note(llm, transcript):
    prompt = f"""
    Transcript:
    {transcript}
    
    Generate a SOAP note:
    Subjective: Patient's complaints.
    Objective: Vital signs and observations.
    Assessment: Diagnosis.
    Plan: Treatment.
    """
    return llm.generate(prompt)
```
