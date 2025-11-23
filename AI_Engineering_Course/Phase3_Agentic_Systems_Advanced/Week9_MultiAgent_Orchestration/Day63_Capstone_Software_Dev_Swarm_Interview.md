# Day 63: Capstone: Building a Software Dev Swarm
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle "Context Contamination" in a Dev Swarm?

**Answer:**
If you feed the Engineer the *entire* codebase every time, you hit context limits.
**Solution:**
*   **RAG for Code:** Index the codebase. When writing `game.py`, retrieve only relevant snippets from `utils.py`.
*   **Skeleton Extraction:** Pass only the *signatures* (class/function definitions) of other files, not the implementation details.

#### Q2: What is the role of the "Product Manager" agent in a technical swarm?

**Answer:**
The PM prevents **Scope Creep** and ensures **Alignment**.
Without a PM, the Architect might over-engineer a "Snake Game" into a "3D Game Engine".
The PM's spec acts as the "Constitution" that the QA agent uses to verify the final product ("Does it actually play Snake?").

#### Q3: How do you secure a Dev Swarm?

**Answer:**
*   **Sandboxing:** Code execution must happen in a disposable Docker container.
*   **Network Block:** The container should have no internet access (unless `pip install` is explicitly allowed/whitelisted).
*   **Resource Limits:** Cap CPU/RAM to prevent infinite loops from crashing the host.

#### Q4: Explain "Test-Driven Development" (TDD) with Agents.

**Answer:**
1.  **QA Agent** writes the test file `test_game.py` *first*, based on the spec.
2.  **Engineer Agent** writes `game.py`.
3.  **Runner** runs `pytest`.
4.  **Loop:** Engineer fixes code until `pytest` passes.
This is often more reliable than "Write then Test" because the test defines the success criteria objectively.

### Production Challenges

#### Challenge 1: The "ImportError" Hell

**Scenario:** Engineer writes `import numpy` but `numpy` isn't installed.
**Root Cause:** Environment mismatch.
**Solution:**
*   **Environment Agent:** A dedicated agent that manages `requirements.txt`. It scans the code, detects imports, and installs them (or adds them to the list).

#### Challenge 2: Code Style & Consistency

**Scenario:** File A uses `snake_case`. File B uses `camelCase`.
**Root Cause:** Independent agents with different biases.
**Solution:**
*   **Linter Tool:** Run `black` or `flake8` automatically on every file save.
*   **Style Guide:** Inject a "Style Guide" into the System Prompt of every Engineer.

#### Challenge 3: Integration Bugs

**Scenario:** `utils.py` works. `game.py` works. But `game.py` calls `utils.helper()` with the wrong arguments.
**Root Cause:** Interface mismatch.
**Solution:**
*   **Type Checking:** Run `mypy` (Static Analysis) to catch type errors across files before running the code.

#### Challenge 4: Cost

**Scenario:** Generating a small app takes 50 steps and $10 in tokens.
**Root Cause:** Inefficient loops.
**Solution:**
*   **Caching:** If the Engineer is rewriting the same function, cache the result.
*   **Human Handoff:** If the agent fails 3 times, ping a human dev to fix the one line of code blocking the swarm.

### System Design Scenario: AI-Powered IDE Plugin

**Requirement:** A VS Code extension that implements "Feature X" for the user.
**Design:**
1.  **Context:** The plugin reads the currently open file and related files (using LSP - Language Server Protocol).
2.  **Plan:** It proposes a plan ("I will modify `auth.py` and create `login.html`").
3.  **Diff:** It generates a *Diff*, not the full file (to save tokens and latency).
4.  **Review:** The user accepts/rejects the Diff.

### Summary Checklist for Production
*   [ ] **Sandboxing:** Mandatory for execution.
*   [ ] **Linter:** Auto-format code.
*   [ ] **TDD:** Write tests before code.
*   [ ] **Dependency Graph:** Build files in the right order.
*   [ ] **Cost Cap:** Stop after N dollars.
