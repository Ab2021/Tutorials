# Lab 27.2: Golden Paths (Software Templates)

## 🎯 Objective

ClickOps for Developers. Instead of copying an old repo and finding/replacing "old-name" with "new-name", developers should use a **Template**. This ensures every new service starts with best practices (Linting, Dockerfile, CI/CD) built-in.

## 📋 Prerequisites

-   Backstage running (Lab 27.1).
-   GitHub Token (with Repo scope) configured in `app-config.yaml`.

## 📚 Background

### Scaffolder
The engine that runs templates.
-   **Parameters**: Inputs form the user (Name, Owner).
-   **Steps**: Actions to take (Fetch template, Rename files, Publish to GitHub, Register in Catalog).

---

## 🔨 Hands-On Implementation

### Part 1: Define the Template 📄

1.  **Create `template.yaml`:**
    In a `templates/react-app` folder in your repo.
    ```yaml
    apiVersion: scaffolder.backstage.io/v1beta3
    kind: Template
    metadata:
      name: create-react-app-template
      title: Create React App
      description: Create a new React application
    spec:
      owner: web-team
      type: website

      parameters:
        - title: App Details
          required:
            - name
          properties:
            name:
              title: Name
              type: string
              description: Unique name of the app
            owner:
              title: Owner
              type: string
              description: Owner of the component
              ui:field: OwnerPicker
              ui:options:
                allowedKinds:
                  - Group

      steps:
        - id: fetch-base
          name: Fetch Base
          action: fetch:template
          input:
            url: ./content
            values:
              name: ${{ parameters.name }}

        - id: publish
          name: Publish
          action: publish:github
          input:
            allowedHosts: ['github.com']
            description: This is ${{ parameters.name }}
            repoUrl: github.com?owner=your-user&repo=${{ parameters.name }}

        - id: register
          name: Register
          action: catalog:register
          input:
            repoContentsUrl: ${{ steps.publish.output.repoContentsUrl }}
            catalogInfoPath: '/catalog-info.yaml'
    ```

### Part 2: The Content (Skeleton) 💀

1.  **Create `content/` folder:**
    Inside `templates/react-app/content`.
    Add `package.json`:
    ```json
    {
      "name": "${{ values.name }}",
      "version": "1.0.0"
    }
    ```
    Add `catalog-info.yaml`:
    ```yaml
    apiVersion: backstage.io/v1alpha1
    kind: Component
    metadata:
      name: ${{ values.name }}
    spec:
      type: website
      owner: ${{ values.owner }}
    ```

### Part 3: Register Template 📝

1.  **Register:**
    Same as Lab 27.1. Register the `template.yaml`.

### Part 4: Run It 🏃‍♂️

1.  **Create:**
    -   Go to **Create**.
    -   Select **Create React App**.
    -   Fill form: Name `my-new-app`, Owner `team-a`.
    -   Click **Next** -> **Create**.

2.  **Verify:**
    -   Backstage creates a new Repo in your GitHub.
    -   It replaces `${{ values.name }}` with `my-new-app`.
    -   It registers it in the Catalog.

---

## 🎯 Challenges

### Challenge 1: Add CI/CD (Difficulty: ⭐⭐⭐)

**Task:**
Add a `.github/workflows/ci.yaml` to the `content/` folder.
Now every app created from this template automatically has a working CI pipeline!

### Challenge 2: Custom Action (Difficulty: ⭐⭐⭐⭐)

**Task:**
Write a custom Scaffolder Action (Node.js).
Example: `action: send:slack`.
Send a message to Slack when a new app is created.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Just add the file. The `fetch:template` action copies everything recursively and processes templating syntax.
</details>

---

## 🔑 Key Takeaways

1.  **Standardization**: If you update the template, all *new* apps get the update. (Updating *existing* apps is harder - requires "One-Click Update" plugins).
2.  **Velocity**: Devs don't spend 2 days setting up Webpack/Jest. They start coding business logic in 5 minutes.
3.  **Governance**: You can enforce that every app has a `CODEOWNERS` file and a `README.md`.

---

## ⏭️ Next Steps

We are building apps fast. Are we spending too much money?

Proceed to **Module 28: Cost Optimization (FinOps)**.
