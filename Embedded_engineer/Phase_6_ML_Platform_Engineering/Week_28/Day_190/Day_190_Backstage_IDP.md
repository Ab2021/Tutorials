# Day 190: The Golden Path: Building an IDP with Backstage
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 28: Platform Engineering Practices

---

> **🎯 Focus Area:** Your 50 Data Scientists shouldn't be writing Kubernetes YAML. They shouldn't even *know* what an Ingress is. They should click "Create New Model", select "PyTorch", and get a URL. **Backstage** (by Spotify) is the UI for your Platform.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** Backstage using the connection to your Git repository.
2.  **Define** the Software Catalog (`catalog-info.yaml`) to track ownership of ML Services.
3.  **Create** a Scaffolder Template that generates a Hello-World PyTorch Repo + Helm Charts + CI/CD Pipeline.
4.  **Integrate** TechDocs to render Markdown documentation next to the service.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `npx @backstage/create-app`.
- Github/Gitlab Token.

---

## 📖 Theoretical Foundation

### 1. Cognitive Load
Platform Engineering is about reducing Cognitive Load.
*   **Max load:** "Here is a raw AWS account. Good luck."
*   **Min load:** "Click this button. You have an API."
*   **The Golden Path:** The supported, opinionated way to build software. You *can* go off-road, but the paved road is faster.

### 2. Backstage Core
*   **Catalog:** Database of all Component, API, Resource, and User entities.
*   **Scaffolder:** Template engine (Cookiecutter) that creates new Git repos.
*   **TechDocs:** Documentation as Code (MkDocs).
*   **Plugins:** Kubernetes, CircleCI, ArgoCD, Prometheus integrations.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: The Catalog Entity

Every repository needs this file in the root.

#### 📁 `catalog-info.yaml`
```yaml
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: image-classifier
  description: Serving ResNet50 for Product tagging.
  tags:
    - python
    - pytorch
    - ml
  annotations:
    github.com/project-slug: myorg/image-classifier
    backstage.io/techdocs-ref: dir:.
spec:
  type: service
  lifecycle: production
  owner: data-science-team
  system: recommendation-engine
```

### 👨‍💻 Core Implementation: The Scaffolder Template

Automating project creation.

#### 📁 `templates/pytorch-service/template.yaml`
```yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: pytorch-template
  title: PyTorch Inference Service
  description: Creates a FastAPI app wrapping a PyTorch model with Docker and Helm.
spec:
  owner: platform-team
  type: service
  
  # 1. Inputs (Form)
  parameters:
    - title: Service Details
      required:
        - name
      properties:
        name:
          title: Name
          type: string
          description: Unique name of the component
        python_version:
          title: Python Version
          type: string
          default: "3.9"
          enum: ["3.8", "3.9", "3.10"]

  # 2. Steps (Actions)
  steps:
    - id: fetch-base
      name: Fetch Skeleton
      action: fetch:template
      input:
        url: ./skeleton # Path to Jinja2 files
        values:
          name: ${{ parameters.name }}
          
    - id: publish
      name: Publish to GitHub
      action: publish:github
      input:
        allowedHosts: ['github.com']
        description: This is ${{ parameters.name }}
        repoUrl: 'github.com?repo=${{ parameters.name }}&owner=myorg'

    - id: register
      name: Register in Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps.publish.output.repoContentsUrl }}
        catalogInfoPath: '/catalog-info.yaml'
```

### 👨‍💻 Core Implementation: The Skeleton

The files that get copied.

#### 📁 `templates/pytorch-service/skeleton/main.py`
```python
from fastapi import FastAPI
# Jinja2 templating happens here
app = FastAPI(title="{{ values.name }}")

@app.get("/")
def healhz():
    return {"status": "ok"}
```

---

## 🔬 Lab Exercise: "Zero to Hero"

### Task
Onboard a new Junior DS.
1.  **Scenario:** Junior DS needs to deploy a model.
2.  **Old Way:** Read Wiki, copy-paste YAML, fail, ask Senior, wait 2 days.
3.  **New Way (IDP):**
    *   Log in to Backstage.
    *   Click "Create". Select "PyTorch Service".
    *   Enter Name: "cat-detector". Owner: "junior".
    *   Click "Next".
4.  **Result:**
    *   GitHub Repo created.
    *   CI Pipeline running and deploying to `dev` cluster.
    *   ArgoCD App created.
    *   DS gets a URL: `https://cat-detector.dev.myorg.com`.
    *   Time taken: **5 minutes**.

---

## 📖 Advanced Theory: InnerSource
Backstage encourages InnerSource.
Since all services are in the Catalog, Team A can discovery Team B's API easily.
*   **Search:** "Does anyone have a 'PDF Parser'?"
*   **Docs:** Read Team B's API docs in TechDocs.
*   **Access:** Request access via Plugin.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Product Mindset:** The Platform is a Product. The Developers are your Customers. If the IDP is hard to use, they won't use it.
2.  **Plugins:** The power of Backstage is the ecosystem. Use the `Kubernetes` plugin to show Pod status directly in the Service page. Use `Cost Insights` to show the bill.
3.  **Skeleton Quality:** The Templates must be production-ready. They must include Linting, Testing, Dockerfile best practices, and Security Scanning by default.

### API Summary
```bash
yarn dev # Start Backstage
```

---

**Day 190 Complete** ✅

*Next: Day 191 - Policy as Code - The Guardrails.*
