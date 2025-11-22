# Lab 22.1: ArgoCD ApplicationSets

## 🎯 Objective

Scale GitOps. Creating 1 `Application` YAML is fine. Creating 50 is toil. **ApplicationSets** allow you to generate Applications automatically based on a list, a Git directory structure, or Cluster secrets.

## 📋 Prerequisites

-   ArgoCD installed (from Module 13).

## 📚 Background

### Generators
-   **List Generator**: Iterate over a fixed list (e.g., `cluster-a`, `cluster-b`).
-   **Git Generator**: Iterate over folders in a Git repo.
-   **Cluster Generator**: Iterate over all clusters registered in ArgoCD.

---

## 🔨 Hands-On Implementation

### Part 1: The Git Generator 📂

We want to deploy every app found in the `apps/` folder of our repo.

1.  **Repo Structure:**
    Imagine a repo `my-gitops-repo`:
    ```text
    apps/
      guestbook/
        deployment.yaml
      payment/
        deployment.yaml
    ```

2.  **Create `appset-git.yaml`:**
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: ApplicationSet
    metadata:
      name: my-apps
      namespace: argocd
    spec:
      generators:
      - git:
          repoURL: https://github.com/argoproj/argocd-example-apps.git
          revision: HEAD
          directories:
          - path: apps/*
      template:
        metadata:
          name: '{{path.basename}}'
        spec:
          project: default
          source:
            repoURL: https://github.com/argoproj/argocd-example-apps.git
            targetRevision: HEAD
            path: '{{path}}'
          destination:
            server: https://kubernetes.default.svc
            namespace: '{{path.basename}}'
          syncPolicy:
            automated:
              prune: true
              selfHeal: true
            syncOptions:
              - CreateNamespace=true
    ```

3.  **Apply:**
    ```bash
    kubectl apply -f appset-git.yaml
    ```

4.  **Verify:**
    Check ArgoCD UI. You should see multiple apps created automatically (one for each folder in `apps/*`).

### Part 2: The List Generator (Multi-Env) 🌍

Deploy `guestbook` to `dev` and `prod`.

1.  **Create `appset-list.yaml`:**
    ```yaml
    apiVersion: argoproj.io/v1alpha1
    kind: ApplicationSet
    metadata:
      name: guestbook-multienv
      namespace: argocd
    spec:
      generators:
      - list:
          elements:
          - cluster: engineering-dev
            url: https://1.2.3.4
          - cluster: engineering-prod
            url: https://5.6.7.8
      template:
        metadata:
          name: 'guestbook-{{cluster}}'
        spec:
          project: default
          source:
            repoURL: https://github.com/argoproj/argocd-example-apps.git
            targetRevision: HEAD
            path: guestbook
          destination:
            server: '{{url}}'
            namespace: guestbook
    ```
    *Note:* You need valid cluster URLs. For this lab, you can use `https://kubernetes.default.svc` for both and just change the namespace.

---

## 🎯 Challenges

### Challenge 1: Matrix Generator (Difficulty: ⭐⭐⭐⭐)

**Task:**
Combine generators.
Iterate over **Clusters** (Dev, Prod) AND **Apps** (Guestbook, Payment).
*Result:* 2 Clusters * 2 Apps = 4 Applications generated.

### Challenge 2: PR Generator (Difficulty: ⭐⭐⭐)

**Task:**
Research the **Pull Request Generator**.
Configure it to create an "Ephemeral Environment" for every Pull Request opened in your repo.
*Goal:* Preview environments for developers.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Use `matrix` generator.
```yaml
generators:
  - matrix:
      generators:
        - git: ...
        - list: ...
```
</details>

---

## 🔑 Key Takeaways

1.  **Automation**: ApplicationSets are the "Factory" for Applications.
2.  **Self-Service**: With the PR Generator, devs get a preview env automatically just by opening a PR. No Ops intervention needed.
3.  **Monorepo vs Polyrepo**: AppSets work well with Monorepos (Git Generator) and Polyrepos (SCM Provider Generator).

---

## ⏭️ Next Steps

We generated apps. Now let's deploy them safely.

Proceed to **Lab 22.2: Argo Rollouts**.
