# Day 207: IF it isn't Documented, It Doesn't Exist: TechDocs
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 30: Capstone Project Part 2

---

> **🎯 Focus Area:** "How do I use Titan?" If you have to answer this question via Slack, you failed. Documentation must be **Code**, live in the Repo, and render beautifully in the IDP.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** Documentation-as-Code using Markdown and MkDocs (`mkdocs.yml`).
2.  **Integrate** TechDocs into Backstage to render docs automatically.
3.  **Draft** an Architecture Decision Record (ADR) explaining "Why we chose Karmada over KubeFed".
4.  **Create** an On-Call Runbook for "High Latency Alert".

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `mkdocs`, `techdocs-cli`.

---

## 📖 Theoretical Foundation

### 1. Docs as Code
*   **Version Control:** Docs live in Git alongside the code.
*   **Review:** Docs are reviewed in PRs. "Update the docs" is a requirement for merging.
*   **Rendering:** MkDocs converts Markdown -> Static HTML. Backstage serves this HTML.

### 2. The 4 Types of Documentation (Diátaxis)
1.  **Tutorials:** "Learning-oriented". Step-by-step lesson (e.g., "Deploying your first Model").
2.  **How-To Guides:** "Problem-oriented". Solve a specific task (e.g., "How to rotate API keys").
3.  **Reference:** "Information-oriented". Facts (e.g., API Swagger, CLI Flags).
4.  **Explanation:** "Understanding-oriented". Context (e.g., "Why we use Ray").

---

## 💻 Implementation

### 👨‍💻 Infrastructure: MkDocs Setup

Create the documentation structure.

#### 📁 `titan-platform/mkdocs.yml`
```yaml
site_name: Titan Platform
plugins:
  - techdocs-core

nav:
  - Home: index.md
  - Getting Started:
    - Quickstart: getting-started/quickstart.md
    - Architecture: getting-started/architecture.md
  - Guides:
    - Training: guides/training.md
    - Serving: guides/serving.md
  - Runbooks:
    - High Latency: runbooks/latency.md
  - ADRs:
    - 001-Karmada: adrs/001-karmada.md
```

#### 📁 `titan-platform/catalog-info.yaml` (Updated)
```yaml
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: titan-platform
  annotations:
    backstage.io/techdocs-ref: dir:. # Docs are in this folder
spec:
  type: documentation
  owner: platform-team
```

### 👨‍💻 Core Implementation: The Runbook

When PagerDuty wakes you up at 3AM, you read this.

#### 📁 `titan-platform/docs/runbooks/latency.md`
```markdown
# Runbook: High Inference Latency

**Severity:** SEV-2
**Trigger:** `HighLatency` Alert (> 500ms p99).

## 1. Triage
- Check Grafana Dashboard: [Titan / Overview](https://grafana.titan.ai)
- Is it Global or Region specific?
    - **Global:** Check Control Plane / DNS.
    - **Region:** Check Region specific metrics.

## 2. Investigation Steps
1. **Check Saturation:**
   - Are GPUs 100% utilized? -> *Action: Scale Up.*
2. **Check Dependencies:**
   - Is Redis slow? -> *Action: check Redis dashboard.*
3. **Check Logs:**
   - Go to Loki. Query: `{app="inference"} |= "timeout"`.

## 3. Mitigation
- **Rollback:** If a deployment happened < 1 hour ago, rollback.
- **Shed Load:** Enable "Degraded Mode" (Drop non-critical requests).
```

### 👨‍💻 Core Implementation: Architecture Decision Record (ADR)

Document the history of decisions.

#### 📁 `titan-platform/docs/adrs/001-karmada.md`
```markdown
# ADR 001: Use Karmada for Federation

| Status | Date | Authors |
| :--- | :--- | :--- |
| Accepted | 2023-10-01 | Alice, Bob |

## Context
We need to manage clusters in US, EU, and AP. We evaluated KubeFed and Karmada.

## Decision
We will use **Karmada**.

## Consequences
- **Positive:** Karmada supports "Pull Mode", allowing clusters to be behind firewalls (Edge).
- **Negative:** Karmada is newer than KubeFed, potentially less stable.
- **Mitigation:** We will contribute to Karmada upstream to fix bugs.
```

---

## 🔬 Lab Exercise: "Local Preview"

### Task
See what the user sees.
1.  **Command:** `techdocs-cli serve`.
2.  **Output:** Serving at `http://localhost:3000`.
3.  **Action:** Edit `index.md`.
4.  **Result:** Browser hot-reloads.
5.  **Bonus:** Add a Mermaid Diagram to `architecture.md`.
    ```mermaid
    graph LR
    User --> Backstage
    ```
    Verify it renders.

---

## 📖 Advanced Theory: Search
Backstage indexes TechDocs using **Elasticsearch** (or Lunr for small setups).
This means a developer can type "GPU OOM" in the global search bar and find your specific troubleshooting guide instantly.
**Tagging:** Add metadata tags to markdown frontmatter to improve search ranking.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Don't write PDFs:** PDFs are where information goes to die. They are unsearchable, hard to version, and hard to update.
2.  **Code Snippets:** Use `code` blocks with language highlighting. TechDocs supports "Copy to Clipboard" buttons automatically.
3.  **Broken Links:** Use a linter (like `markdown-link-check`) in CI to fail the PR if a doc links to a 404 page.

### API Summary
```bash
mkdocs build
techdocs-cli publish --publisher-type awsS3 --storage-name my-bucket --entity default/component/titan-platform
```

---

**Day 207 Complete** ✅

*Next: Day 208 - The Pitch - Presentation & Demo.*
