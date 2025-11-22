# Lab 7.3: Terraform State & Variables

## 🎯 Objective

Master the "Brain" of Terraform (State) and make your code reusable (Variables). You will learn why the `terraform.tfstate` file is critical and how to parameterize your configurations.

## 📋 Prerequisites

-   Completed Lab 7.2.

## 📚 Background

### The State File (`terraform.tfstate`)
Terraform maps your code (`main.tf`) to real-world resources (`ID: i-12345`).
-   **Purpose**: Tracking metadata, performance caching, and syncing.
-   **Rule**: Never edit it manually.
-   **Risk**: If you delete it, Terraform forgets the resources exist and will try to create them again (or fail to delete them).

### Variables (`variables.tf`)
Hardcoding values (like "my-bucket-name") is bad.
-   **Input Variables**: Parameters passed *into* the module.
-   **Output Values**: Data passed *out* of the module (e.g., IP address).

---

## 🔨 Hands-On Implementation

### Part 1: Variables 📦

1.  **Create `variables.tf`:**
    ```hcl
    variable "filename" {
      default     = "pets.txt"
      description = "Name of the file to create"
      type        = string
    }

    variable "content" {
      default     = "We love animals!"
      type        = string
    }
    ```

2.  **Update `main.tf`:**
    Use `var.name`.
    ```hcl
    resource "local_file" "pet" {
      filename = "${path.module}/${var.filename}"
      content  = var.content
    }
    ```

3.  **Apply with Defaults:**
    ```bash
    terraform apply -auto-approve
    ```
    *Result:* Creates `pets.txt`.

4.  **Apply with Overrides:**
    ```bash
    terraform apply -var "filename=dogs.txt" -var "content=We love dogs" -auto-approve
    ```
    *Result:* Creates `dogs.txt`. (Note: It might destroy `pets.txt` depending on how you structured it, or just track the new one).

### Part 2: Outputs 📤

1.  **Create `outputs.tf`:**
    ```hcl
    output "file_id" {
      value = local_file.pet.id
    }
    ```

2.  **Apply:**
    ```bash
    terraform apply -auto-approve
    ```
    *Result:* At the end, it prints:
    `file_id = "..."` (The SHA hash of the file).

### Part 3: The State File Experiment 🧪

1.  **View State:**
    ```bash
    cat terraform.tfstate
    ```
    *Observe:* It's JSON. It contains the file content and ID.

2.  **Simulate Disaster:**
    Delete the file manually (bypass Terraform).
    ```bash
    rm dogs.txt
    ```

3.  **Run Plan:**
    ```bash
    terraform plan
    ```
    *Result:* Terraform sees the file is missing (Drift Detection) and plans to **recreate** it. `+ resource ...`.

4.  **Simulate State Loss:**
    **Warning:** Only do this in a lab!
    ```bash
    rm terraform.tfstate
    ```

5.  **Run Plan:**
    ```bash
    terraform plan
    ```
    *Result:* Terraform thinks *nothing* exists. It plans to create `dogs.txt`. If `dogs.txt` was already there, the apply might fail (File exists error).

---

## 🎯 Challenges

### Challenge 1: terraform.tfvars (Difficulty: ⭐⭐)

**Task:**
Instead of typing `-var` every time, create a file named `terraform.tfvars`.
Set `filename = "cats.txt"`.
Run `terraform apply`.
*Note:* Terraform automatically loads files ending in `.tfvars`.

### Challenge 2: Sensitive Variables (Difficulty: ⭐⭐⭐)

**Task:**
Mark the `content` variable as `sensitive = true`.
Run `terraform apply`.
Check the Output in the terminal. It should say `(sensitive value)`.
*Check `terraform.tfstate`.* Is it hidden there? (Spoiler: No! State is plain text).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`terraform.tfvars`:
```hcl
filename = "cats.txt"
```

**Challenge 2:**
`variables.tf`:
```hcl
variable "content" {
  sensitive = true
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Don't Commit .tfvars**: If it contains secrets, add `*.tfvars` to `.gitignore`.
2.  **State Locking**: In teams, use Remote State (S3 + DynamoDB) to prevent two people applying at once.
3.  **Drift**: Terraform detects if the real world doesn't match the code.

---

## ⏭️ Next Steps

We know the syntax. Let's provision a real server.

Proceed to **Lab 7.4: Provisioning AWS EC2**.
