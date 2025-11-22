# Lab 1.10: DevOps Career Paths & Skills

## 🎯 Objective

Map out your personal DevOps career journey. You will analyze job descriptions, identify skill gaps, create a learning roadmap, and draft a "DevOps-ready" resume section.

## 📋 Prerequisites

-   Completed Module 1 (Labs 1.1 - 1.9).
-   Honest self-reflection.

## 📚 Background

### The "DevOps Engineer" Title

"DevOps Engineer" is a controversial title (since DevOps is a culture), but it is the industry standard term.

**Common Roles:**
1.  **DevOps Engineer**: Generalist. CI/CD, Cloud, IaC.
2.  **Site Reliability Engineer (SRE)**: Focus on coding, reliability, and scaling (Google model).
3.  **Platform Engineer**: Builds internal developer platforms (IDP) for other devs.
4.  **Cloud Engineer**: Focus on AWS/Azure/GCP architecture.
5.  **DevSecOps Engineer**: Focus on security integration.

**Salary Expectations (2024/2025):**
-   Junior: $80k - $110k
-   Mid-Level: $120k - $160k
-   Senior: $170k - $250k+
-   *Note: Highly dependent on location and company.*

---

## 🔨 Hands-On Implementation

### Part 1: The Skill Matrix Assessment 📊

**Objective:** Quantify your current skills.

1.  **Create `skills_matrix.md`:**
    Copy the table below and rate yourself (0-5).
    0 = No clue, 5 = Can teach it.

    | Category | Skill | Rating (0-5) | Target |
    |----------|-------|--------------|--------|
    | **Culture** | Agile/Scrum | | 3 |
    | | CI/CD Concepts | | 4 |
    | **OS** | Linux CLI | | 4 |
    | | Bash Scripting | | 3 |
    | **Code** | Python/Go | | 3 |
    | | Git | | 4 |
    | **Cloud** | AWS/Azure | | 3 |
    | **IaC** | Terraform | | 3 |
    | **Containers**| Docker | | 4 |
    | | Kubernetes | | 3 |
    | **Observability**| Prometheus/Grafana | | 2 |

2.  **Analyze Gaps:**
    Identify the rows where `Rating < Target`. These are your learning priorities for the rest of this course.

### Part 2: Analyzing the Market 🕵️‍♂️

**Objective:** Find out what companies actually want *today*.

1.  **Research:**
    Go to LinkedIn, Indeed, or specialized tech boards.
    Search for "DevOps Engineer" in your area (or Remote).
    Find 3 job postings:
    -   One "Junior" or "Associate".
    -   One "Senior".
    -   One "Dream Job" (FAANG, etc.).

2.  **Create `market_analysis.md`:**
    Extract the keywords.

    ```markdown
    # Market Analysis

    ## Common Requirements
    - [ ] Docker
    - [ ] AWS
    - [ ] Python
    - [ ] ...

    ## Surprising Requirements
    - [ ] (e.g., "Experience with Mainframes" or "Rust")

    ## Soft Skills
    - [ ] "Communication"
    - [ ] "Problem Solving"
    ```

### Part 3: The Resume Transformation 📄

**Objective:** Rewrite your experience to highlight DevOps impact.

**Scenario:** You are a Sysadmin or Developer transitioning to DevOps.

**The "Before" Bullet Point:**
> "Managed Linux servers and installed updates."

**The "After" (DevOps) Bullet Point:**
> "Automated Linux server patching using Ansible, reducing maintenance window from 4 hours to 15 minutes and eliminating manual errors."

**Exercise:**
Create `resume_draft.md`. Write 3 bullet points from your past experience using the **STAR Method** (Situation, Task, Action, Result), emphasizing **Automation**, **Scale**, or **Reliability**.

1.  *Situation/Task*: "Deployments were manual and took 2 hours."
2.  *Action*: "Implemented a CI/CD pipeline using GitHub Actions."
3.  *Result*: "Reduced deployment time to 5 minutes and increased frequency by 400%."

### Part 4: The Portfolio Project Plan 🏗️

**Objective:** Plan the "Capstone Project" you will build during this course to show recruiters.

**Create `portfolio_plan.md`:**

```markdown
# My DevOps Portfolio Project

## Application
A simple Python/Node.js web app (e.g., a To-Do list or Weather app).

## Infrastructure
- [ ] Hosted on AWS (Free Tier).
- [ ] Provisioned via Terraform (IaC).

## Pipeline
- [ ] Code in GitHub.
- [ ] CI/CD via GitHub Actions.
- [ ] Dockerized application.

## Observability
- [ ] Basic monitoring (Uptime/Logs).

## Documentation
- [ ] A README that explains *how* to run it.
- [ ] An architecture diagram.
```

---

## 🎯 Challenges

### Challenge 1: The Elevator Pitch (Difficulty: ⭐⭐)

**Scenario:** You step into an elevator with the VP of Engineering. They ask: "So, what is DevOps?"

**Task:** Write a 30-second (50 word) answer in `elevator_pitch.txt`.
*Hint: Don't list tools. Talk about value (Speed, Stability, Culture).*

### Challenge 2: The Mock Interview (Difficulty: ⭐⭐⭐)

**Scenario:** Common interview questions.

**Task:** Write answers to these in `interview_prep.md`:
1.  "Explain what happens when you type google.com in a browser (DevOps perspective)."
2.  "What is the difference between a Virtual Machine and a Container?"
3.  "How would you handle a situation where a developer keeps breaking the build?"
4.  "Explain 'Infrastructure as Code' to a non-technical manager."

---

## 💡 Solution Guide

<details>
<summary>Click to reveal Interview Answers</summary>

**1. Google.com flow:**
DNS resolution -> Load Balancer -> SSL Termination -> Web Server (Nginx) -> App Server -> Database. Mention Caching (CDN/Redis).

**2. VM vs Container:**
VM = Virtual Hardware + Full OS (Heavy).
Container = Shared OS Kernel + App Dependencies (Lightweight).
Analogy: House (VM) vs Apartment (Container).

**3. Developer breaking build:**
Not about punishment.
1. Help them debug locally.
2. Improve local testing tools (pre-commit hooks).
3. Implement "Pull Request" gates so they *can't* merge broken code.

**4. IaC for Manager:**
"Instead of clicking buttons in a console (which is slow and error-prone), we write a 'recipe' for our servers. We can give this recipe to the cloud provider to build the exact same infrastructure every time, automatically."
</details>

---

## 🔑 Key Takeaways

1.  **T-Shaped Skills**: You need broad knowledge (the top of the T) and deep knowledge in one area (the vertical bar).
2.  **Impact over Tools**: Recruiters care that you "saved money" or "saved time," not just that you know "Jenkins."
3.  **Always Be Learning**: The tools change every 2 years. The principles last forever.

---

## ⏭️ Next Steps

**Congratulations!** You have completed Module 1: Introduction to DevOps.

You now have the cultural foundation, the toolchain, and the roadmap. It's time to get technical.

Proceed to **Module 2: Linux Fundamentals**.
