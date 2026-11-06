# <Topic Name>

> **Purpose:** [What this cheat sheet specifically derives, computes, selects or diagnoses — not a repetition of its title.]
> **Prerequisites:** [File NN — topic]; know [2–3 concrete concepts].
> **Key outcome:** After reading, you can [testable action] [specific target].
> **What this file does NOT cover:** [2–3 related exclusions; see another file for each].
> **Level:** [1 Foundations | 2 Core Methods | 3 Advanced / Production]
> **Version / reviewed:** [1.0 / YYYY-MM-DD] | **Evidence:** [DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC]
> **Display-math style:** [LaTeX OR fenced plain-text, consistently with INDEX]

**Authoring sequence:** Draft Section 12's Quick Reference Card **first**, then author Sections 1–11. Publish the card **last**. For every formula, state the **validity condition before use**, then its 5-part anatomy and derivation. Use the `Cheat_Sheet_Outline_Template_v3_Integrated.md` master guide for detailed instructions.

---

## Section 1 — File Header [REQUIRED]

**Specific scope:** [One sentence describing precisely what is included.]

**Reader task:** Given [inputs and conditions], the reader will be able to [produce measurable result or decision].

**Explicit exclusions / next reads:** [2–3 topics not covered and a real linked file for each.]

**Takeaway:** [One sentence.]

## Section 2 — Concept Map / Mental Model [REQUIRED]

*One-screen orientation BEFORE the first formula. Choose one of two formats.*

```text
[Input / concept]
    |
    v
[Transformation] ---- [Constraint / assumption]
    |
    v
[Output / decision]
```

| Concept / method | Core idea | Equation ID | Use when |
|---|---|---|---|
| [concept] | [...] | [Eq. 1] | [...] |

**Takeaway:** [Why this organization helps.]

## Section 3 — Foundational Definitions [REQUIRED]

### [Term 1]

- **Definition:** [formal, unambiguous]
- **Intuition:** [analogy/plain-English restatement]
- **Notation:** [symbol; units; shape and domain]

### [Term 2]

- **Definition:** [...]
- **Intuition:** [...]
- **Notation:** [...]

| Symbol | Meaning | Units / dimension | Domain | INDEX location |
|---|---|---|---|---|
| `[symbol]` | [...] | [...] | [...] | [glossary row] |

**Takeaway:** [Minimal insight required for Section 4.]

## Section 4 — Core Formulas with Full Derivation [REQUIRED]

### 4.1 Simplest / Base Case

**Validity first:** This formula applies when [conditions]. Do NOT apply when [violating scenario].

**Derive before stating:**

1. [Starting definition / equality.]
2. [Algebraic step and the assumption that permits it.]
3. [Result; distinguish exact equality from approximation.]

**[1] FORMULA:**

```text
[canonical display formula in the collection's selected math style]
```

**[2] WHERE:**

| Symbol | Meaning | Units / shape |
|---|---|---|
| [symbol] | [...] | [...] |

**[3] CONSTRAINT:** [when valid, assumptions, bounds, numerical stability requirements]

**[4] NUMERICAL EXAMPLE:** [inputs → substitution → answer with units]

**[5] INTERPRETATION:** [what the answer tells the practitioner]

### 4.2 General Case

[Repeat five-part anatomy; explain how it reduces to base case.]

### 4.3 Special Cases / Variants

| Variant | Formula change | Assumptions | Use when | Avoid when |
|---|---|---|---|---|
| [variant] | [...] | [...] | [...] | [...] |

### 4.4 Limiting Behaviour / Counterexample

- **Limit:** [n→∞ / h→0 / parameter→boundary]
- **Violation:** [what goes wrong when a condition fails]
- **Practical effect:** [what action changes]

**Takeaway:** [One sentence.]

## Section 5 — Worked Numerical Example [REQUIRED]

### Worked Example: [Descriptive title]

- **Data:** [origin or ILLUSTRATIVE label; units and dimensions]
- **Goal:** [target result]
- **Inputs / notation:** [values; symbol definitions]

| Step | Formula | Substitute | Intermediate | Rounded answer | Sanity check |
|---|---|---|---|---|---|
| 1 | [...] | [...] | [...] | [...] | [...] |
| 2 | [...] | [...] | [...] | [...] | [...] |

**Final result:** [quantity with units and interpretation]

**Independent check:** [unit check, limit or recomputation]

**Wrong path:** [violated assumption → incorrect result → corrected method]

**Production-scale difference:** [classroom-sized parameter and real/estimated realistic range, clearly labeled]

**Takeaway:** [One sentence.]

## Section 6 — Visual / Geometric Intuition [OPTIONAL; REQUIRED for mathematical proofs and matrix operations]

```text
        [accurately labeled axes, quantities and key points]
```

**Annotated key point:** [name/value]

**Caption / intuitive takeaway:** [one sentence explaining the mathematical relationship]

## Section 7 — Comparison / Selection Table [REQUIRED when ≥2 methods are covered]

| Method / variant | Assumption | Complexity | Best for | Avoid when |
|---|---|---|---|---|
| A | [...] | [...] | [...] | [nonblank] |
| B | [...] | [...] | [...] | [nonblank] |

```text
IF [condition] AND [constraint] -> [method] BECAUSE [reason]
ELSE IF [condition]           -> [alternative]
ELSE                          -> [fallback / collect more data]
```

**Comparable empirical benchmark (if available):**

| Variant | Exact workload | Metric | Result | Source / year | Evidence status |
|---|---|---|---|---|---|
| [method] | [...] | [...] | [measured or NOT MEASURED] | [...] | sourced / measured / illustrative |

**Takeaway:** [One-line decision rule.]

## Section 8 — Failure Modes & Edge Cases [REQUIRED for every sheet]

*Minimum four substantive rows: one cause, test and corrective action per row.*

| Symptom | Underlying cause | Diagnostic check | Correct action |
|---|---|---|---|
| [failure 1] | [...] | [...] | [...] |
| [failure 2] | [...] | [...] | [...] |
| [failure 3] | [...] | [...] | [...] |
| [failure 4] | [...] | [...] | [...] |

**Topic-specific cases:** [Choose from source Section 8's six categories: statistical, matrix, optimization, probability, hardware/systems, agents/LLMs.]

**Failure trace:** Trigger → observable symptom → evidence → root cause → fix → prevention.

**Add to final Common Issues file:** [ISSUE ID; provenance OBSERVED / REPRODUCED / HYPOTHETICAL].

**Takeaway:** [One sentence.]

## Section 9 — Complexity & Cost Analysis [REQUIRED for algorithmic/systems topics; OPTIONAL for statistical theory]

### 9a. Algorithmic Complexity

| Operation | Time | Space | Notes / assumptions |
|---|---|---|---|
| Training / fitting | `O(?)` | `O(?)` | [bottleneck] |
| Single inference | `O(?)` | `O(?)` | [dominant term] |
| Update / online learning | `O(?)` | `O(?)` | [or N/A with reason] |

### 9b. Back-of-Envelope Sizing

- **Workload / version:** [input size, data type, device and software]
- **Formula and units:** [e.g., FLOPs, bytes, tokens, operations, dollars]
- **Numeric substitution:** [show calculation]
- **Constraint:** [device capability, bandwidth, concurrency]
- **Result / bottleneck:** [quantity and bottleneck]
- **Assumption sensitivity:** [how conclusion changes at a different scale]
- **Textbook vs production:** [small demo vs measured/sourced/illustrative deployed range]

**If not relevant:** `N/A — [specific mathematical reason]`.

**Takeaway:** [One sentence.]

## Section 10 — Key Theorems / Results Reference [OPTIONAL; REQUIRED for mathematical theory sheets]

| Theorem / result | Statement (brief but correct) | Essential assumptions | Why it matters | Source |
|---|---|---|---|---|
| [theorem] | [...] | [...] | [practical implication] | [textbook/paper] |

**Boundary / counterexample:** [How overstating the theorem would mislead.]

**Takeaway:** [What the theorem allows you to conclude, and no more.]

## Section 11 — Interview / Exam Questions [OPTIONAL; for exam-prep collections]

*Include 10–15 questions if enabled, distributed across three levels. Put all answers here, before Section 12.*

### Level 1 — Definition / Recall

1. **Q:** [definition question] **A:** [checked short answer]
2. **Q:** [...] **A:** [...]

### Level 2 — Derivation / Apply

3. **Q:** [derive or compute] **A:** [step-by-step calculation]
4. **Q:** [...] **A:** [...]

### Level 3 — Judgement / Design / Debug

5. **Q:** [identify failure and choose alternative] **A:** [reasoned solution]
6. **Q:** [...] **A:** [...]

**Coverage / missing questions:** [Complete to 10–15 when section is enabled.]

**Takeaway:** [One sentence.]

## Section 12 — Quick Reference Card [REQUIRED — MUST BE THE LAST SECTION]

*Draft first as an outline. Fill and publish after Sections 1–11. Keep this 5–10-row table as the final content of this individual cheat sheet.*

| Item | Value |
|---|---|
| Topic | [file topic] |
| Core formula | [one most important correct equation] |
| Key assumption | [assumption most commonly violated] |
| Complexity | [dominant operation: time, memory or N/A] |
| Use when | [one-line selection criterion] |
| Avoid when | [one-line anti-pattern] |
| Go deeper | [actual next numbered file and topic] |
