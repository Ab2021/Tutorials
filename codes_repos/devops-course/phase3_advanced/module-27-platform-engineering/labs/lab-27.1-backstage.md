# Lab 27.1: Backstage (Internal Developer Platform)

## 🎯 Objective

Stop the spreadsheet chaos. **Backstage** (by Spotify) is an open platform for building developer portals. It provides a centralized **Software Catalog** to track ownership, metadata, and dependencies of all your microservices.

## 📋 Prerequisites

-   Node.js (Active LTS) installed.
-   Yarn installed.
-   Docker installed.

## 📚 Background

### Core Features
-   **Software Catalog**: "Who owns `payment-service`?"
-   **Software Templates**: "Create a new Spring Boot app in 5 clicks."
-   **TechDocs**: "Docs like Code."
-   **Plugins**: Integrate with CircleCI, Kubernetes, ArgoCD, Jira.

---

## 🔨 Hands-On Implementation

### Part 1: Create the App 🏗️

1.  **Scaffold:**
    ```bash
    npx @backstage/create-app@latest
    ```
    *Prompts:*
    -   Name: `my-backstage-app`
    -   Database: `SQLite` (for testing)

2.  **Start:**
    ```bash
    cd my-backstage-app
    yarn dev
    ```

3.  **Access:**
    Open `http://localhost:3000`.
    You see the default catalog with some sample components.

### Part 2: Register a Component 📝

1.  **Create `catalog-info.yaml`:**
    In your own GitHub repo (e.g., `my-service`), create this file:
    ```yaml
    apiVersion: backstage.io/v1alpha1
    kind: Component
    metadata:
      name: my-service
      description: A sample service for Lab 27.1
      annotations:
        github.com/project-slug: your-user/my-service
    spec:
      type: service
      lifecycle: experimental
      owner: team-a
      system: payment-system
    ```

2.  **Register:**
    -   In Backstage UI, click **Create**.
    -   Click **Register Existing Component**.
    -   Paste the URL to your `catalog-info.yaml` (Raw GitHub URL).
    -   Click **Analyze** -> **Import**.

3.  **Verify:**
    Go to **Catalog**. Search for `my-service`.
    Click it. You see the ownership and metadata.

### Part 3: TechDocs 📄

1.  **Add Docs:**
    In your repo, create a `docs/` folder and an `index.md`.
    Add `backstage.io/techdocs-ref: dir:.` to your `catalog-info.yaml` annotations.

2.  **View:**
    In Backstage, click the **Docs** tab on your component.
    *Result:* Your markdown is rendered beautifully inside the portal.

---

## 🎯 Challenges

### Challenge 1: Kubernetes Plugin (Difficulty: ⭐⭐⭐)

**Task:**
Enable the Kubernetes plugin.
Configure it to connect to your local Minikube.
*Goal:* See the Pod status of `my-service` directly inside Backstage.

### Challenge 2: CI/CD Integration (Difficulty: ⭐⭐)

**Task:**
Add the `github-actions` annotation.
`github.com/project-slug: user/repo`
*Result:* See the latest build status in the CI/CD tab.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Requires configuring `app-config.yaml` with the cluster details and a ServiceAccount token.
</details>

---

## 🔑 Key Takeaways

1.  **Discoverability**: New joiners can find "How to run the payment service" in seconds.
2.  **Ownership**: Every service must have an owner. No more orphan code.
3.  **Extensibility**: Backstage is just a React app. You can write your own plugins.

---

## ⏭️ Next Steps

We have a catalog. Now let's enable Self-Service.

Proceed to **Lab 27.2: Software Templates**.
