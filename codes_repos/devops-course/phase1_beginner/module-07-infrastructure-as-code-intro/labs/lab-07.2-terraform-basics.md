# Lab 7.2: Terraform Basics (Providers & Resources)

## 🎯 Objective

Write your first Terraform code. To avoid cloud costs and complexity immediately, we will use the **Local Provider** to manage files on your laptop. This teaches the core HCL syntax safely.

## 📋 Prerequisites

-   Completed Lab 7.1.

## 📚 Background

### HCL (HashiCorp Configuration Language)
The syntax of Terraform.
```hcl
<BLOCK TYPE> "<BLOCK LABEL>" "<NAME>" {
  key = "value"
}
```

### Core Concepts
1.  **Provider**: The plugin that talks to the API (e.g., `aws`, `local`, `google`).
2.  **Resource**: The thing you want to create (e.g., `aws_instance`, `local_file`).
3.  **Attribute**: Properties of the resource (e.g., `ami`, `filename`).

---

## 🔨 Hands-On Implementation

### Part 1: The Setup 📂

1.  **Create a directory:**
    ```bash
    mkdir terraform-lab-1
    cd terraform-lab-1
    ```

2.  **Create `main.tf`:**
    This is the default entry point.

### Part 2: The Local Provider 📄

We will tell Terraform to create a text file.

1.  **Edit `main.tf`:**
    ```hcl
    # 1. Configure the Provider
    terraform {
      required_providers {
        local = {
          source  = "hashicorp/local"
          version = "~> 2.0"
        }
      }
    }

    # 2. Define a Resource
    resource "local_file" "pet" {
      filename = "${path.module}/pet.txt"
      content  = "We love DevOps!"
    }
    ```

### Part 3: The Workflow (Init, Plan, Apply) 🔄

1.  **Init (Initialize):**
    Downloads the provider plugins.
    ```bash
    terraform init
    ```
    *Output:* `Terraform has been successfully initialized!`

2.  **Plan (Preview):**
    Shows what will happen.
    ```bash
    terraform plan
    ```
    *Output:* `+ resource "local_file" "pet" { ... }` (Green + means create).

3.  **Apply (Execute):**
    Makes it happen.
    ```bash
    terraform apply
    ```
    *Prompt:* Type `yes`.
    *Result:* `Apply complete! Resources: 1 added`.

4.  **Verify:**
    Check your folder. `pet.txt` exists!
    ```bash
    cat pet.txt
    ```

### Part 4: The Power of IaC (Updates) ⚡

1.  **Modify `main.tf`:**
    Change the content.
    ```hcl
    resource "local_file" "pet" {
      filename = "${path.module}/pet.txt"
      content  = "We love DevOps and Terraform!"
    }
    ```

2.  **Apply:**
    ```bash
    terraform apply -auto-approve
    ```
    *Output:* `~ update in-place`. (Yellow ~ means modify).

3.  **Verify:**
    `cat pet.txt` -> Content changed.

### Part 5: Destroy 💥

1.  **Cleanup:**
    ```bash
    terraform destroy -auto-approve
    ```
    *Result:* `pet.txt` is deleted.

---

## 🎯 Challenges

### Challenge 1: Multiple Files (Difficulty: ⭐⭐)

**Task:**
Add a second resource to `main.tf` that creates `secrets.txt` with content "SuperSecret".
Run `terraform apply`.
*Note:* You don't need to run `init` again (same provider).

### Challenge 2: Random Provider (Difficulty: ⭐⭐⭐)

**Task:**
1.  Add the `random` provider to `required_providers`.
2.  Create a resource `random_pet` "my_pet"`.
3.  Update the file content to be: `content = "My pet is ${random_pet.my_pet.id}"`.
4.  Run `init` (new provider!) and `apply`.
    *Result:* You get a random name like "funny-zebra".

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```hcl
resource "local_file" "secret" {
  filename = "${path.module}/secrets.txt"
  content  = "SuperSecret"
}
```

**Challenge 2:**
```hcl
resource "random_pet" "my_pet" {
  length = 2
}

resource "local_file" "pet" {
  filename = "${path.module}/pet.txt"
  content  = "My pet is ${random_pet.my_pet.id}"
}
```
</details>

---

## 🔑 Key Takeaways

1.  **Workflow**: `Init` -> `Plan` -> `Apply` -> `Destroy`. Memorize this.
2.  **State File**: Look at `terraform.tfstate`. This is where Terraform stores its brain. **NEVER** delete it manually.
3.  **Interpolation**: `${...}` allows you to reference other resources (e.g., putting a random ID into a file).

---

## ⏭️ Next Steps

We managed local files. Now let's manage the Cloud.

Proceed to **Lab 7.3: Terraform State & Variables**.
