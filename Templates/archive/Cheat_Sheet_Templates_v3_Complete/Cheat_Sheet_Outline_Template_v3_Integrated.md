# Cheat Sheet Writing Guide & Outline Template

> **For technical domains: math-heavy, formula-first, less code.** Synthesized from: CUDA H100, LLM Finetuning, Fraud Modelling, Time Series Forecasting, Agentic AI, LangGraph/LangChain, AI-DS, and additional collections whose names are cropped in the screenshot.
>
> **Integrated edition (v3.0, November 2026):** Updated from **all 19 supplied screenshots**. Parts A–E, Sections 1–12, all ten photographed improvement tips, the original collection checklist and collection skeleton are now included. Original material is kept distinct from paragraphs labeled **Editorial extension**. This version expands the source into a quality-gated, reusable master guide. For image-faithful reconstruction, use `Cheat_Sheet_Outline_Template_v3_Source_Faithful.md`.


## Guide-wide authoring contract — editorial extension

Write a cheat sheet that helps a practitioner **derive, compute, diagnose, select, and recall**, not merely define. Every nontrivial claim must be either (a) derived from stated assumptions, (b) numerically checked, (c) sourced to an authoritative reference, or (d) labeled as a teaching simplification. The default is **formula-first, prose-light, implementation-light**; code may be added when it verifies a mathematical result or clarifies a diagnostic.

**Evidence labels:** `DERIVED` (mathematical consequence), `VERIFIED` (tested calculation), `SOURCED` (citable third-party fact), `ILLUSTRATIVE` (constructed example), `HEURISTIC` (experience-based rule), `OPEN` (unresolved/uncertain). Avoid treating heuristic thresholds as universal truths.

**What a great sheet must answer:** What is it? Why does it work? What assumptions are hidden? How do I calculate it? When does it fail? Which alternative should I use? How can I verify and recall it quickly?

**Source of truth:** One working file per focused topic, index links by relative path, stable notation across related sheets, and no fabricated benchmark numbers or citations.

---

## How to Use This Template

1. **Read the Collection-Level Rules** first — they apply to every file you write.
2. For each individual cheat sheet file, follow the **Per-File Outline (Sections 1–12)**.
3. Mark each section **[REQUIRED]** or **[OPTIONAL]** per the tags below before drafting.
4. After all files are written, write the **INDEX file last** (Section 0).
5. Apply the **10 Improvement Tips** as a revision pass before finalising.
6. Run the **Minimum Viable Checklist** before marking the collection complete.

---

# Part A — Collection-Level Rules (Apply Before Writing Any File)

## A1. Naming Convention

```text
00_INDEX.md                          ← always present, always first
01_<Topic_Name>.md                   ← zero-padded two-digit prefix
02_<Topic_Name>.md
...
NN_<N>_Common_Issues.md              ← always last; 25 or 50 issues minimum
```

**Rules:**

- File names use underscores, `Title_Case` topic words.
- Topics ordered from foundational to advanced (not alphabetically).
- Number before topic lets readers follow a learning path without reading the index.
- Common Issues file is always the final file; it makes the collection practical.

## A2. Topic Ladder (Required Before Writing File 01)

Before writing a single file, map your topics on a three-level ladder.

| Level | Label | What it covers | Example (CUDA) | Example (Time Series) |
|---|---|---|---|---|
| Foundations | Level 1 | Definitions, notation, prerequisites | GPU execution hierarchy, memory types | Stationarity, ACF/PACF, white noise |
| Core Methods | Level 2 | Key algorithms, formulas, derivations, comparisons | WGMMA tile math, TMA descriptors | ARIMA family, ETS state-space |
| Advanced / Production | Level 3 | Tradeoffs, edge cases, failure modes, composability | Multi-GPU topology, roofline model | Hierarchical reconciliation, MLOps monitoring |

**Write files in ladder order. Never put a Level 3 concept in File 01.**

## A3. Mathematical Notation Standard

Pick one of two styles for the entire collection and never mix:

| Style | When to use | Example |
|---|---|---|
| LaTeX block (`$$...$$`) | When the audience renders Markdown with MathJax (Jupyter, Obsidian, GitHub with plugins) | `$$\hat{y}_{t+h} = \mu + \phi^h (y_t - \mu)$$` |
| Code-block math (plain text in triple-backtick fences) | When rendering is unknown or the audience is a practitioner who reads raw Markdown | `y_hat(t+h) = mu + phi^h * (y_t - mu)` |

Annotate every symbol immediately after the formula. Never introduce a Greek letter or hat notation without defining it.

## A4. Formula Block Anatomy (Required for Every Formula)

Every formula block must have these five parts, in order:

```text
[1] FORMULA          — the equation itself
[2] WHERE            — symbol definitions table or list
[3] CONSTRAINT       — when the formula is valid / assumptions required
[4] NUMERICAL EXAMPLE — a worked case with real numbers
[5] INTERPRETATION   — one sentence on what the result tells you
```

**Example — ARIMA forecast error variance:**

```text
[1] FORMULA
Var(ŷ_{T+h}|T) = sigma^2 * SUM_(j=0)^(h-1) psi_j^2

[2] WHERE
sigma^2 = residual variance (estimated from training residuals)
psi_j   = MA(inf) coefficient at lag j (from ARMA inversion)
h       = forecast horizon

[3] CONSTRAINT
Valid only when the process is invertible and stationary.
For non-stationary (ARIMA with d>0), apply after differencing.

[4] NUMERICAL EXAMPLE
AR(1) with phi=0.8, sigma^2=1.0, h=3:
  psi_0 = 1, psi_1 = 0.8, psi_2 = 0.64
  Var = 1.0 * (1^2 + 0.8^2 + 0.64^2) = 1.0 * (1 + 0.64 + 0.41) = 2.05
  95% interval width = ±1.96 * sqrt(2.05) = ±2.80

[5] INTERPRETATION
Forecast uncertainty grows with horizon; for AR(1) it converges to
sigma^2 / (1 - phi^2) as h -> infinity.
```

> **Transcription note:** The example's approximations (`0.64²` rounded to `0.41`, interval reported as `±2.80`) mirror the screenshot rather than silently replacing its numbers.

## A5. Table Types Reference

Use the right table for the right purpose. Never explain in prose what a table can show.

| Table type | Purpose | Mandatory columns |
|---|---|---|
| Comparison matrix | Rank or select between N methods | Method \| Metric A \| Metric B \| Use when |
| Formula glossary | Define all symbols in one place | Symbol \| Meaning \| Units \| Typical range |
| Decision guide | Steer the reader to the right choice | Condition \| Recommended choice \| Why |
| Error / issue table | Common issues file | Symptom \| Root cause \| Diagnosis command \| Fix |
| Complexity table | Algorithm scaling | Operation \| Time complexity \| Space complexity \| Notes |
| Tradeoff matrix | Compare design decisions | Option \| Strength \| Weakness \| When to pick |
| Empirical benchmark | Ground results in evidence | Condition \| Metric \| Source / year |

## A6. Index File (File 00) — Write Last

The INDEX file is the instrument panel for the whole collection. Write it after all other files exist.

**Required sections in the INDEX file:**

1. One-paragraph overview of what the collection covers and what it assumes.
2. Topic Ladder table (Level 1 / 2 / 3 mapping to file numbers).
3. Quick Topic Lookup table (keyword → file number).
4. Reading paths: Beginner path / Practitioner path / Advanced path.
5. Prerequisites (what to know before opening file 01).
6. What this collection does **NOT** cover (scope boundary).


## A7. Collection Metadata, Source Quality & Maintenance — editorial extension

Every sheet should begin with a small machine-readable record. This is an **added recommendation**, not a field visible in the photographs.

```yaml
---
collection: "[collection_slug]"
file_id: "01"
topic: "[Topic Name]"
level: 1              # 1 foundations | 2 core methods | 3 advanced/production
status: draft         # draft | reviewed | published | deprecated
version: "1.0"
created: YYYY-MM-DD
reviewed: YYYY-MM-DD
review_trigger: "[version, release, benchmark, or conceptual dependency that would invalidate claims]"
notation_style: latex # latex | plain-text
prerequisites: ["00_INDEX.md"]
pattern_tags: ["[controlled_tag]"]
source_status: "[DERIVED | VERIFIED | SOURCED | ILLUSTRATIVE | HEURISTIC | OPEN]"
---
```

**Notation registry:** Record each symbol's meaning, units, dimension/shape, domain, and first appearance. If the same symbol legitimately has different meanings in adjacent sheets, qualify it or create a glossary warning. Use a single display-math style per collection, but inline notation need not be identical to fenced display code. Never present a rounded value as exact.

**Index planning reconciles the original rules:** Create `00_INDEX.md` with a topic ladder and provisional links **before** drafting, as a design checklist; populate, test, and **finalize it last** after filenames are fixed. This preserves the original “write INDEX last” rule while avoiding an unplanned collection.

**Citations and benchmarks:** Include source title/link, publication year, exact workload, hardware/software version, metric definition and measurement conditions where relevant. Use `NOT MEASURED` if evidence is absent. Disclose whether examples use synthetic numbers.

---

# Part B — Per-File Outline (Sections 1–12)

Each cheat sheet file is one Markdown document covering one focused topic. Target length: **800–1,800 words per file (excluding tables and formula blocks)**.

## Section 1 — File Header [REQUIRED]

```markdown
# <Topic Name>

> **Purpose:** One sentence on what this sheet covers.
> **Prerequisites:** <File NN — topic>. Know: <2-3 concepts the reader must already understand>.
> **Key outcome:** After reading, you will be able to [action verb] [specific thing].

---
```

**Source Tip 9 implemented — scope boundary:** Add to the header: `> **This file does NOT cover:** [2–3 excluded topics]. See [relative link] for [related topic].` Keep the excluded-topic link valid.

**Source Tip 10 implemented — write the Quick Reference Card first:** Draft the Section 12 card as an *outline before* writing Sections 2–11, and place the finished card **last** in the published file. Check that every card row is justified in the main content.

**Source Tip 9 implemented — scope boundary:** Add to the header: `> **This file does NOT cover:** [2–3 excluded topics]. See [relative link] for [related topic].` Keep the excluded-topic link valid.

**Source Tip 10 implemented — write the Quick Reference Card first:** Draft the Section 12 card as an *outline before* writing Sections 2–11, and place the finished card **last** in the published file. Check that every card row is justified in the main content.

**Rules:**

- Purpose is not a restatement of the title — it adds what angle or depth.
- Prerequisites create an explicit dependency chain between files.
- Key outcome is a testable skill, not a vague goal ("calculate X" not "understand X").

**Examples across collections:**

| Collection | Good purpose line |
|---|---|
| CUDA | “H100 hardware specs, SM execution hierarchy, memory subsystem, tensor core generations.” |
| Time Series | “Master Box-Jenkins ARIMA — identify, estimate, diagnose, and forecast with the full ARIMA family.” |
| Fraud | “Supervised model selection guide, WoE/IV scorecard mathematics, threshold calibration.” |
| Agents | “Four core agentic patterns, their cost models, stopping criteria, and composition rules.” |

### Header acceptance test — editorial extension

- Include the exact **topic boundary** (what is excluded), skill verb (derive / compute / diagnose / choose), level, source status and `last reviewed` date.
- A prerequisite must point to an existing file, never a conceptual circular dependency.
- If the sheet has a numerical result, explicitly identify its units and approximation convention.
- Include a one-sentence retrieval cue: “If the problem mentions **[symptom]**, start at Section **[number]**.”

## Section 2 — Concept Map / Mental Model [REQUIRED]

Before any formulas, give the reader a one-screen orientation.

**Two acceptable formats:**

**Format A — ASCII hierarchy (for systems, architectures, pipelines):**

```text
Concept A
├── Sub-concept 1 (description)
│   ├── Detail 1a
│   └── Detail 1b
└── Sub-concept 2 (description)
    └── Detail 2a
```

**Format B — Comparison table (for methods, models, options):**

| Concept | Core idea | Key equation | Use when |
|---|---|---|---|
| AR(p) | Past values predict future | `y_t = Σ phi_j * y_{t-j} + ε` | Autocorrelated, stationary |
| MA(q) | Past errors predict future | `y_t = μ + Σ θ_j * ε_{t-j}` | Short memory, mean-reverting |
| ARMA(p,q) | Both | Combines above | General stationary series |

**Rule:** The concept map must appear on screen before the first formula. It is the reader's anchor.

### Model/map validity check — editorial extension

Specify the **inputs → transformation → outputs**, relevant layers or variables, and at least one important dependency arrow. An ASCII sketch must be mathematically/architecturally correct, not merely visually appealing. When using a comparison map, identify the invariant, changing assumption, and decision boundary; avoid mixing incomparable model classes.

## Section 3 — Foundational Definitions [REQUIRED]

One subsection per key term. Each definition has three parts:

```markdown
### Term Name
- **Definition:** [formal, unambiguous]
- **Intuition:** [one analogy or plain-English restatement]
- **Notation:** [how it appears in formulas]
```

**How many definitions?** Cover only terms the reader must know to follow the rest of the file. Typically **3–8 definitions** per file. If you need more, split the file.

**Source Tip 5 implemented — collection symbol glossary:** A3's symbol-definition rule is local to the sheet; a **master glossary** in `00_INDEX.md` adds `Symbol | Meaning | File(s)` across the entire collection. Where symbols have units/shapes, record these too.

**Source Tip 5 implemented — collection symbol glossary:** A3's symbol-definition rule is local to the sheet; a **master glossary** in `00_INDEX.md` adds `Symbol | Meaning | File(s)` across the entire collection. Where symbols have units/shapes, record these too.

### Definition quality gate — editorial extension

| Term | Mathematical meaning | Shape / units | Valid domain | Closely confused with |
|---|---|---|---|---|
| `[term]` | `[unambiguous definition]` | `[scalar / vector / unit]` | `[constraint]` | `[nearby concept]` |

Provide necessary and sufficient conditions when available, not loose analogies. Attach at least one **nonexample** when a definition is easy to misapply.

## Section 4 — Core Formulas with Full Derivation [REQUIRED]

The mathematical heart of the file. Use Formula Block Anatomy (see A4) for every formula.

**Sub-section structure:**

```text
4.1 Simplest / Base Case
    ➜ Formula Block (5-part anatomy)
    ➜ Stationarity / validity conditions

4.2 General Case (extend from base)
    ➜ Formula Block
    ➜ How the general case reduces to base case

4.3 Key Special Cases or Variants
    ➜ Side-by-side comparison table of variants
    ➜ Decision guide: which variant to use when

4.4 Asymptotic / Limiting Behaviour
    ➜ What happens as n → ∞, h → 0, or the key parameter reaches a boundary
    ➜ Why this matters practically
```

**Formula writing rules:**

- Build from simple to general: derive AR(1) before AR(p), scalar before matrix form.
- Number every equation if you will reference it later (e.g., `Eq. 3.2`).
- Always show at least one negative/boundary case (what happens when a constraint is violated).
- If two formulas look similar, include a “differences table” to contrast them.

**Source Tips 1 & 4 implemented — condition + derivation:** First state the validity condition **before** the formula. Then show the short derivation (at least the immediately preceding algebra step), and finally display the original five-part formula anatomy. The condition may be restated in part [3] so the self-contained block remains valid. Separate `=` from `≈` and label any approximation.

**Mathematical correction for the source ARIMA example:** The variance of the future **prediction error**, not the already-known conditional point forecast, is the object represented by `σ² Σ ψ_j²`. Under an appropriate linear-process/innovations model it is

```text
Var(Y_{T+h} − Y_hat_{T+h|T} | information through T)
    = sigma^2 * SUM_{j=0}^{h-1}(psi_j^2)
```

The photographed `Var(ŷ_{T+h}|T)` notation would be misleading if interpreted literally. Use correct conditioning and define the process being forecast when writing actual technical sheets.

**Source Tips 1 & 4 implemented — condition + derivation:** First state the validity condition **before** the formula. Then show the short derivation (at least the immediately preceding algebra step), and finally display the original five-part formula anatomy. The condition may be restated in part [3] so the self-contained block remains valid. Separate `=` from `≈` and label any approximation.

**Mathematical correction for the source ARIMA example:** The variance of the future **prediction error**, not the already-known conditional point forecast, is the object represented by `σ² Σ ψ_j²`. Under an appropriate linear-process/innovations model it is

```text
Var(Y_{T+h} − Y_hat_{T+h|T} | information through T)
    = sigma^2 * SUM_{j=0}^{h-1}(psi_j^2)
```

The photographed `Var(ŷ_{T+h}|T)` notation would be misleading if interpreted literally. Use correct conditioning and define the process being forecast when writing actual technical sheets.

### Proof obligations and reproducibility — editorial extension

For each major formula, show a compact derivation chain with numbered transformations. At every change, specify which **identity, theorem, approximation, or independence assumption** was used. Distinguish exact identities `=` from approximations `≈`; state sample vs. population, conditional vs. unconditional, and row/column conventions when applicable. If a proof is outside scope, give a short derivation sketch **and an authoritative source**, rather than implying it was proven.

Use this reusable table for hidden assumptions:

| Equation | Necessary assumption | How to check | Fails when | Alternative |
|---|---|---|---|---|
| Eq. `[n]` | `[condition]` | `[diagnostic]` | `[counterexample]` | `[robust choice]` |

**Numerical accuracy note (correcting precision, not the original screenshot):** For the AR(1) example above, `1 + 0.8² + 0.64² = 2.0496`, and `1.96 × √2.0496 ≈ 2.81` when rounded to two decimal places. The photographed `±2.80` is an approximate displayed value, not a reproducible two-decimal rounding. The forecast variance assumes independent innovations, a known process specification and conditional treatment of estimated parameters; parameter-estimation uncertainty would change a real forecast interval. Check the chosen normal critical value is appropriate to the example.

## Section 5 — Worked Numerical Example [REQUIRED]

A single end-to-end calculation showing every formula from Section 4 applied to real numbers.

**Template:**

```markdown
### Worked Example: [Descriptive title]

**Setup:**
  Dataset:  [describe the data, dimensions, context]
  Goal:     [what are we computing?]
  Notation: [restate key symbols for this example]

**Step 1 — [Name]:**
  [formula applied] = [substituted values] = [result]

**Step 2 — [Name]:**
  [formula applied] = [result]

...

**Final result:**
  [quantity with units and plain-English meaning]

**Sanity check:**
  [back-of-envelope check or unit check confirming the result is plausible]
```

**Rules:**

- Use realistic (not toy) numbers where possible — real orders of magnitude.
- Show the sanity check explicitly; it teaches dimensional reasoning.
- If a calculation requires a table, embed the full table (never truncate mid-table).
- For ML/stats formulas: include at least one “wrong path” — show what happens if you use the wrong formula or violate an assumption.

**Source Tip 6 implemented — textbook vs production:** Record **both** a small calculation that can be checked by hand and a production-scale scenario (realistic `n`, feature dimension, batch/sequence length, failure/scale conditions). Label production ranges as measured, sourced, assumed or illustrative. Do not suggest that illustrative “real-world values” are universal.

**Source Tip 6 implemented — textbook vs production:** Record **both** a small calculation that can be checked by hand and a production-scale scenario (realistic `n`, feature dimension, batch/sequence length, failure/scale conditions). Label production ranges as measured, sourced, assumed or illustrative. Do not suggest that illustrative “real-world values” are universal.

### Worked-example audit trail — editorial extension

| Step | Equation / rule | Input values + units | Unrounded intermediate | Reported result + units | Independent check |
|---|---|---|---|---|---|
| 1 | Eq. `[n]` | `[values]` | `[value]` | `[rounded value]` | `[units / substitution]` |

- Record a deterministic input table; specify source or label it `ILLUSTRATIVE`.
- Carry full precision through intermediate steps, and round once at presentation.
- Check **dimensions, sign, limiting behavior, expected order of magnitude**, and at least one independently computed value.
- Demonstrate a meaningful wrong path or violated assumption (without implying that an invalid result is correct).
- If possible, include a 3–10-line **verification snippet** or calculator-friendly substitution; teaching does not require a full code implementation.

## Section 6 — Visual / Geometric Intuition [OPTIONAL — required for mathematical proofs and matrix operations]

When algebra alone is not enough, add a geometric or graphical explanation.

**Acceptable formats:**

**ASCII geometry:**

```text
Loss surface (convex):

     Loss
      |     *
      |  *     *
      |*           *
      +------ theta ------
         ^ global minimum
```

**ASCII comparison (before/after):**

```text
Before normalization:       After normalization:
  Feature A: 0–100,000        Feature A: 0.0–1.0
  Feature B: 0.0–1.0          Feature B: 0.0–1.0
  → gradient dominated        → balanced gradient
    by Feature A                descent
```

**Rules:**

- ASCII art only — no image attachments.
- Always label axes, variables, and key points.
- Include the intuitive takeaway as a sentence below every diagram.

> **Transcription note:** The ASCII geometry is preserved as an illustrative sketch from the source. Its pictured point arrangement is not a precise convex loss curve and should be redrawn before use in a teaching cheat sheet.

**Source Tip 7 implemented — explicit ASCII labeling:** Every illustration needs axis/dimension names, at least one named/valued key point, and a caption. Check that the sketched geometry correctly expresses its mathematical claim. The source convex-loss ASCII example is only schematic and should be redrawn for publication.

**Source Tip 7 implemented — explicit ASCII labeling:** Every illustration needs axis/dimension names, at least one named/valued key point, and a caption. Check that the sketched geometry correctly expresses its mathematical claim. The source convex-loss ASCII example is only schematic and should be redrawn for publication.

### Diagram quality gate — editorial extension

Before publishing, verify that axes and arrows match the mathematical claim, labels are legible in a monospace font, and the drawing has an explicit takeaway. Where the original example's dot arrangement cannot convey a proper convex bowl, prefer a correctly labeled U-shape or a textual description of convexity. Use accessible text descriptions even for ASCII-only diagrams.

## Section 7 — Comparison / Selection Table [REQUIRED when ≥ 2 methods are covered]

A single decision table that answers: “which one should I use?”

| Method / Variant | Assumption | Complexity | Best for | Avoid when |
|---|---|---|---|---|
| A | … | O(n²) | … | … |
| B | … | O(n log n) | … | … |

Then a selection guide in code-block format:

```text
IF interpretability required AND regulatory context  → Method A
IF large N (>1M rows) AND latency < 10ms             → Method B
IF non-stationary AND structural breaks present       → Method C
IF multivariate AND cointegration likely              → Method D
```

**Rules:**

- Every row in the comparison table must have a filled “Avoid when” — never leave it blank.
- The selection guide must be exhaustive: cover the most common decision conditions.
- If more than 6 methods, split into two tables: “Foundational Methods” and “Advanced Methods”.

**Source Tips 2 & 3 implemented — reject invalid options and substantiate comparisons:** Include `Avoid when` for every option. If giving numerical performance comparisons, cite the original workload/protocol, source, date and whether results were reproduced; otherwise show `NOT VERIFIED`. Do not reuse the photographed HumanEval percentages as established facts without an attributable source.

**Source Tips 2 & 3 implemented — reject invalid options and substantiate comparisons:** Include `Avoid when` for every option. If giving numerical performance comparisons, cite the original workload/protocol, source, date and whether results were reproduced; otherwise show `NOT VERIFIED`. Do not reuse the photographed HumanEval percentages as established facts without an attributable source.

### Comparison evidence and fallback — editorial extension

Every option should be evaluated under a **common workload, assumptions and metric definition**. Distinguish asymptotic time complexity from observed latency and memory. Explain selection priority when conditions conflict; show a fallback if no option meets the constraints.

| Decision criterion | Must-have or nice-to-have? | Evidence | Disqualifying condition | Fallback |
|---|---|---|---|---|
| `[criterion]` | `[priority]` | `[measurement/source]` | `[condition]` | `[alternative]` |

For benchmark-based recommendations, record environment, hardware, versions, sample size, uncertainty and date. Do not treat fabricated A/B/C/D selection rules as validated production guidance.

## Section 8 — Failure Modes & Edge Cases [REQUIRED for every sheet]

The section most often skipped — and the one that makes the collection useful in production.

| Symptom | Underlying cause | Diagnostic check | Correct action |
|---|---|---|---|
| [Observable problem] | [Root cause in terms of the math] | [How to confirm] | [What to do] |

**Editorial extension — required diagnostic depth:** For each real failure include **trigger → symptom → test → confirmed cause → correction → prevention**. The photographed final checklist calls for *at least four filled failure-mode rows per sheet*. These should be actual investigated faults or reproducible, explicitly labeled teaching scenarios, not invented anecdotes.

**Required edge cases per topic type:**

| Topic type | Edge cases to always cover |
|---|---|
| Statistical models | Near-unit-root, overfitting (p too large), heteroskedasticity violation |
| Matrix / linear algebra | Rank deficiency, ill-conditioned matrix (high condition number), numerical underflow |
| Optimization / gradient | Saddle points, learning rate too high/too low, vanishing/exploding gradients |
| Probability / distributions | Zero-probability events, degenerate distributions, boundary parameter values |
| Hardware / systems | Memory bandwidth saturation, occupancy collapse, synchronization deadlock |
| Agents / LLMs | Context window overflow, tool-call loops, hallucination in reasoning chain |

**Editorial extension — domain-specific diagnostic checks:**

| Domain | Extra failure condition | How to verify |
|---|---|---|
| Time series | Structural break, target leakage, seasonal lag error | Rolling-origin evaluation, data timestamp audit, residual ACF |
| Fraud / insurance ML | Delayed labels, class imbalance, base-rate shift | Label maturity, PR-AUC/precision at review capacity, segment calibration |
| CUDA | Incorrect tile/stride/layout, race or misalignment | Kernel correctness tests, compute-sanitizer, profiler counters |
| Agentic LLMs | Prompt injection, unbounded tool access, missed termination | Permission logs, adversarial tests, explicit step/budget ceilings |
| Streaming systems | Late events, recovery duplication, backpressure | Replay and checkpoint tests, event-time/lag measurements |

**Source Tip 8 implemented — Common Issues provenance:** In the collection's final troubleshooting file, label each item as one of: violated mathematical assumption, numerical edge case, scale failure, or conceptual confusion. Add provenance (`OBSERVED`, `REPRODUCED`, or `HYPOTHETICAL`) so a constructed test is not falsely presented as an incident. Include the diagnostic and expected outcome.

## Section 9 — Complexity & Cost Analysis [REQUIRED for algorithmic and systems topics; OPTIONAL for statistical theory]

Two sub-sections:

### 9a. Algorithmic Complexity

| Operation | Time | Space | Notes |
|---|---|---|---|
| Training / fitting | O(?) | O(?) | Key bottleneck: [state it] |
| Single inference | O(?) | O(?) | Dominant term at scale: [state it] |
| Update / online learning | O(?) | O(?) | If applicable |

### 9b. Back-of-Envelope Sizing

Always include at least one worked sizing calculation relevant to the topic:

```text
Example — CUDA kernel arithmetic intensity:
  FLOPs per element:       2 (multiply + add per MAC)
  Bytes read per element:  2 (FP16 = 2 bytes)
  Arithmetic intensity:    2 FLOPs / 2 bytes = 1.0 FLOP/byte
  H100 roofline:           memory bound if AI < 134 FLOP/byte → YES, memory bound

Example — ARIMA model training time:
  n = 50,000 observations
  p = 3, q = 2 (ARIMA(3,1,2))
  OLS pass: O(n * (p+q)^2) = O(50000 * 25) = 1.25M operations
  → trivially fast; estimation dominated by optimisation iterations (~50), not n
```

**Editorial extension — size correctly before publishing:** The screenshot's `134 FLOP/byte` cutoff is a scenario-dependent *roofline illustration*, **not** a general constant for an H100. Compute `peak_compute_throughput / sustained_memory_bandwidth` for the selected precision, hardware and workload. Count all actual reads/writes in the arithmetic-intensity denominator, including outputs and re-reads as appropriate. The pictured `2 FLOPs / 2 bytes` is a simplified example, not automatically the true kernel intensity.

The screenshot's `O(n(p+q)^2)` “OLS pass” should **not** be claimed as the full training complexity of ARIMA maximum-likelihood fitting. Record the actual estimation algorithm, optimizer iterations, per-iteration work, series length, parameter counts and numerical implementation. Big-O, FLOP counts, wall-clock timings and dollar cost are different quantities.

**Source Tip 6 applied:** Pair each classroom-sized example with production scale, hardware/software versions, memory, throughput/latency assumptions, and units.

## Section 10 — Key Theorems / Results Reference [OPTIONAL — required for mathematical theory sheets]

For cheat sheets that cover theoretical foundations, list the results compactly:

| Theorem / Result | Statement (brief) | Assumption | Why it matters |
|---|---|---|---|
| Wold's Decomposition | Any covariance-stationary process = deterministic + MA(∞) | Stationarity | Justifies ARMA representation |
| Cramér–Rao Bound | No unbiased estimator has lower variance than 1/I(θ) | Regularity conditions | Sets the floor for estimation quality |
| Universal Approximation | MLP with ≥1 hidden layer approximates any continuous function | Bounded activation, sufficient width | Justifies deep learning expressiveness |

**Rule:** State what the theorem *implies for practice*, not just what it says mathematically.

**Editorial extension — theorem precision:** Use the photographed table only as a mnemonic. For a teaching-quality mathematical sheet, add the formal statement, conditions and counterexample to over-broad interpretations. For example, Wold's decomposition separates deterministic and purely nondeterministic components; the scalar Cramér–Rao bound applies under the appropriate regularity conditions; and universal-approximation guarantees depend on the activation, domain and approximation notion. Include original textbook/paper citations for formal theorem statements.

## Section 11 — Interview / Exam Questions [OPTIONAL — include if collection is for exam prep]

**10–15 questions per file, categorised by difficulty:**

```text
**Level 1 — Definition (recall):**
Q: Define the autocorrelation function (ACF) for a stationary process.

**Level 2 — Derivation (apply formula):**
Q: Derive the variance of a 3-step ahead forecast for AR(1) with phi=0.7, sigma^2=2.

**Level 3 — Judgement (design/debug):**
Q: Your ARIMA model gives an ACF of residuals where lag 12 is significant.
   What does this indicate and what model extension would you consider?
```

**Editorial extension — exam review quality:** If this optional section is present, include questions across recall, application and design/debug, with **short checked answers** or an answer-key link. Avoid treating residual autocorrelation at one lag as a certain diagnosis without considering multiplicity and data context. Keep any self-test **before** Section 12 so that the Quick Reference Card remains the last section.

## Section 12 — Quick Reference Card [REQUIRED]

The last thing in every file. A **5–10 row summary table** a reader can scan in 30 seconds.

**Source Tip 10 implemented — author first, publish last:** Draft this card as the first planning artifact, then verify its cells point to material covered in Sections 1–11. The final card must be placed at the *end* of the file. A compact card should expose the most important decision, assumption and failure condition, not every detail.

| Item | Value |
|---|---|
| Topic | [file topic] |
| Core formula | [the single most important equation] |
| Key assumption | [the assumption most often violated] |
| Complexity | [O(?) for the dominant operation] |
| Use when | [one-line decision rule] |
| Avoid when | [one-line anti-pattern] |
| Go deeper | [file number and topic for the natural next read] |

---

# Part C — 10 Improvement Tips (original photographed guidance)

These are the most common gaps observed across all seven reference collections. Apply as a revision pass after the first draft is complete.

## Tip 1 — Derive Before You State

**Problem:** Cheat sheets often state a formula without showing where it comes from. **Fix:** For every formula introduced from Section 4, show the previous line of algebra that led to it. Even a 3-step derivation (not a full proof) is enough to anchor the formula in the reader's memory.

**Before:** `Var(AR(1)) = sigma^2 / (1 - phi^2)`

**After:**

```text
Var(y_t) = Var(phi * y_{t-1} + epsilon_t)
         = phi^2 * Var(y_{t-1}) + Var(epsilon_t)   [independence of epsilon]
         = phi^2 * Var(y_t) + sigma^2              [stationarity: Var(y_t) = Var(y_{t-1})]
Solving: Var(y_t) * (1 - phi^2) = sigma^2
         Var(y_t) = sigma^2 / (1 - phi^2)
```

## Tip 2 — Every Comparison Table Needs an “Avoid When” Column

**Problem:** Comparison tables tell readers what a method is good for but not when to disqualify it. **Fix:** Add an “Avoid when” or “Watch out for” column to every comparison table. Deciding **NOT** to use a method is as important as knowing when to use it.

## Tip 3 — Use Empirical Benchmarks to Anchor Abstract Claims

**Problem:** Claims like “X outperforms Y” or “this method is fast” are unmemorable without numbers. **Fix:** Include at least one empirical benchmark row per comparison table.

**Example (Agents collection):**

| Condition | HumanEval Score |
|---|---:|
| GPT-3.5, single pass | 48.1% |
| GPT-3.5, agentic workflow | 95.1% |

This one row is more memorable than a paragraph explaining agentic workflows.

## Tip 4 — State Constraints Before the Formula, Not After

**Problem:** Readers apply formulas before reading the assumption footnote at the bottom. **Fix:** Put the stationarity condition, invertibility requirement, or validity range **BEFORE** the formula block. Structure: “This formula is valid when X. Under that condition, [formula].”

## Tip 5 — Build a Notation Glossary in the INDEX File

**Problem:** Each file defines its own notation. Readers moving between files lose track of symbols. **Fix:** Maintain a master symbol glossary in `00_INDEX.md`.

| Symbol | Meaning | File(s) |
|---|---|---|
| phi | AR coefficient | 02, 03 |
| sigma² | Noise variance | 02, 03, 05 |
| d_model | Embedding dimension | 07, 08 |

This costs one table in the INDEX and saves confusion throughout the collection.

## Tip 6 — Distinguish “Textbook” from “Production” Values

**Problem:** Formulas use clean textbook numbers (n=10, p=2). Production reality differs. **Fix:** Add a “Real-world range” column or note in the complexity and sizing sections.

**Example:**

```text
Textbook example: AR(2) with n=20, easy to see by hand
Production reality: n = 2M observations, p selected via AIC up to p=15,
                    multivariate (10 series), rolling re-estimation every week
```

The delta between textbook and production is what gets practitioners stuck.

## Tip 7 — Label Every ASCII Diagram Explicitly

**Problem:** ASCII diagrams are drawn but axes, variables, and key points are not labelled. **Fix:** Every ASCII diagram must have:

- Axis labels (or dimension labels for hierarchies).
- Annotation of at least one key point with its value or name.
- A caption sentence below the diagram.

## Tip 8 — Write the “Common Issues” File from Real Errors, Not Theory

**Problem:** Common Issues files are often written by imagining errors, not from real failure patterns. **Fix:** Every issue in the final file should be traced to one of:

- A formula applied with a violated assumption.
- A numerical edge case (near-zero, infinity, rank deficiency).
- A scale problem (textbook code that breaks on 10M rows).
- A conceptual confusion between two similar terms.

If you cannot trace an issue to one of these four categories, it probably does not belong.

## Tip 9 — Add “What This File Does NOT Cover” to Every Prerequisites Block

**Problem:** Readers enter a cheat sheet not knowing its scope boundary and get confused when a topic they expected is missing. **Fix:** Add one sentence at the end of every file header:

> This file does NOT cover: [list 2–3 related topics explicitly excluded]. See file NN for [excluded topic A].

## Tip 10 — Reverse-Engineer the Quick Reference Card First

**Problem:** Writers draft the full file and then write a Quick Reference Card as a summary. The result is a card that summarises what was written, not what matters most. **Fix:** Write the Quick Reference Card **BEFORE** drafting the full file. Use it as a compass: if a row in the card does not appear in the main text, the main text is incomplete. If a section of the main text does not contribute to any row in the card, that section may be unnecessary.

---

## Integration map — how the original 10 tips become enforceable requirements (editorial)

| Original tip | Integrated requirement | Publication test |
|---|---|---|
| 1. Derive before stating | Section 4 shows the algebra leading to the result | Each important formula has derivation evidence |
| 2. Avoid when | Section 7 comparison includes disqualifiers | No option lacks an `Avoid when` |
| 3. Empirical benchmarks | Section 7/9 comparisons include sourced context | No naked performance percentages |
| 4. Constraints first | A4/Section 4 put a validity warning before the equation | Assumption is visible before use |
| 5. Master notation glossary | A3/INDEX track every symbol and its locations | Symbols reconcile across files |
| 6. Textbook vs production | Sections 5/9 show both small and realistic scale | Units, dimensions and scale explicit |
| 7. Label ASCII | Section 6 labels axes, key points, caption | Diagram intelligible without nearby prose |
| 8. Real Common Issues | Section 8/final issues file require diagnostic provenance | ≥25 unique issues, each diagnosable |
| 9. Excluded scope | Section 1 identifies 2–3 things not covered | Reader knows where to go next |
| 10. Card-first drafting | Section 12 card drafted first; placed last | Card/main-text claims match |

# Part D — Minimum Viable Cheat Sheet Collection Checklist (original + validation extensions)

Run this before marking the collection complete.

### Collection level

- [ ] `00_INDEX.md` exists with topic ladder, quick lookup, and three reading paths.
- [ ] Files are numbered sequentially and ordered foundation → advanced.
- [ ] Common Issues file is the last file and has ≥25 issues.
- [ ] Mathematical notation style is consistent across all files (LaTeX or code-block, not both).
- [ ] Master symbol glossary exists in INDEX or as a standalone file.

### Per-file level (check each file individually)

- [ ] File header has Purpose, Prerequisites, Key outcome.
- [ ] Concept Map appears before the first formula.
- [ ] Every formula has all 5 parts: formula, where, constraint, numerical example, interpretation.
- [ ] At least one Worked Numerical Example using realistic (not toy) numbers.
- [ ] Comparison table has “Avoid when” column.
- [ ] Failure Modes table present with at least 4 rows.
- [ ] Complexity / sizing analysis present (or explicitly noted as N/A with reason).
- [ ] Quick Reference Card is the last section in every file.
- [ ] No section ends without a concluding one-line takeaway.

### Content quality

- [ ] Every claim is supported by a formula, a table, or an empirical number.
- [ ] No symbol appears without being defined on its first use.
- [ ] Every “best practice” is paired with the failure mode it prevents.
- [ ] The collection can be read in ladder order (file 01 → 02 → ... → NN) with no forward dependencies.
- [ ] A practitioner could use the Quick Reference Cards alone to make a reasonable decision.

---

### Additional validation gates — editorial extension

- [ ] Every numerical comparison is **sourced / reproduced / explicitly illustrative** (no unattributed benchmark interpreted as a fact).
- [ ] Formulas show correct conditioning, domains, units and stable notation; important derived identities are recomputed independently.
- [ ] Mathematical theorem statements name essential hypotheses or link to precise, verifiable versions.
- [ ] Complexity claims distinguish algorithmic asymptotics from wall-clock runtime and real hardware bottlenecks.
- [ ] The final Common Issues file distinguishes observed/reproduced incidents from hypothetical examples.
- [ ] Every question in Section 11 (when used) has a validated answer or answer-key reference.
- [ ] Markdown fences and tables render, and all links, file numbers and prerequisite chains resolve.

# Part E — Collection Skeleton (Fill In for a New Topic)

Use this to plan a new cheat sheet collection before writing any content.

```text
Collection name: __________________________________________
Domain: __________________________________________________
Target audience: __________________________________________
Prerequisites (what should reader know first): ____________

Topic Ladder:
  Level 1 — Foundations:
    File 01: __________________
    File 02: __________________

  Level 2 — Core Methods:
    File 03: __________________
    File 04: __________________
    File 05: __________________

  Level 3 — Advanced / Production:
    File 06: __________________
    File 07: __________________
    File 08: __________________

  Final file (NN): 50 Common Issues — ______________

Master symbols to include in INDEX glossary:
  Symbol | Meaning
  ______ | ___________________
  ______ | ___________________

Reading paths:
  Beginner (3–4 files): files __ → __ → __ → __
  Practitioner (6–7 files): files __ → __ → __ → __ → __ → __ → __
  Advanced (all files): 01 → NN

What this collection does NOT cover: _______________________
Recommended pre-reading: __________________________________
Recommended next collection after completing this one: _____
```

---


## Collection authoring workflow — editorial extension

1. **Plan:** Fill Part E, define Level 1/2/3 sequence and a controlled glossary; create a draft `00_INDEX.md` navigation stub.
2. **Design each sheet backwards:** Draft its seven-field Quick Reference Card and measurable outcome before the body.
3. **Build the evidence:** Define symbols, show the derivations, calculate a worked example, check a failure and record selection tradeoffs.
4. **Scale and verify:** Write Sections 8–10 with diagnostics, boundaries, production-relevant sizing and exact assumptions.
5. **Review rigor:** Check original 10 improvements, math/source integrity, and the Part D checklist.
6. **Publish:** Place the final Quick Reference Card at the end of each sheet, create ≥25 distinct Common Issues, resolve relative links and finalize the INDEX **last**.

## Technical cautions when adapting the photographed examples — editorial extension

- **HumanEval benchmark**: The Tip 3 values (48.1% and 95.1%) do not show an attributable protocol/source in the photo. Keep as a photographed illustration, not verified performance evidence.
- **Roofline**: Do not reuse a single `134 FLOP/byte` intensity boundary for all H100 precisions and workloads.
- **ARIMA complexity**: An illustrative OLS pass is not a validated total training-time formula for all ARIMA estimators.
- **Theorem precision**: Source Section 10 abbreviates important hypotheses; use full, defensible theorem statements in the final teaching material.
- **Consistency**: Drafting the INDEX structure early and *finalizing* it last reconciles the original A6 and Part E. Place a validity condition before a formula, but preserve the five-part formula anatomy within the block for standalone reference.
