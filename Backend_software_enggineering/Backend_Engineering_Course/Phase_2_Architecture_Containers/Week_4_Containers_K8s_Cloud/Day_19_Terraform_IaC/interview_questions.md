# Day 19: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is storing Terraform State in Git a bad idea?
**Answer:**
1.  **Secrets**: The state file contains raw data (including DB passwords) in plain text.
2.  **Locking**: Git doesn't support locking. Two engineers running `apply` at the same time will corrupt the state.
3.  **Solution**: Use a Remote Backend (S3 with Encryption) + DynamoDB (for Locking).

### Q2: What is "Drift" and how does Terraform handle it?
**Answer:**
*   **Drift**: When the real infrastructure changes outside of Terraform (e.g., someone manually deleted a server).
*   **Detection**: When you run `terraform plan`, Terraform refreshes the state (queries AWS) and compares it to your code.
*   **Resolution**: It will propose actions to fix the drift (e.g., recreate the deleted server).

### Q3: Explain `count` vs `for_each`.
**Answer:**
*   **count**: Creates N copies.
    *   `count = 3` -> `instance[0]`, `instance[1]`.
    *   *Risk*: If you remove index 0, Terraform shifts index 1 to 0, forcing a recreate.
*   **for_each**: Creates copies based on a map/set.
    *   `for_each = { web = "t2.micro", api = "t2.large" }`.
    *   *Benefit*: Stable IDs. Removing "web" doesn't affect "api".

---

## Scenario-Based Questions

### Q4: You have an existing AWS S3 bucket created manually. How do you bring it under Terraform control?
**Answer:**
*   **Command**: `terraform import`.
*   **Steps**:
    1.  Write the `resource "aws_s3_bucket" "b" {}` block in code.
    2.  Run `terraform import aws_s3_bucket.b my-existing-bucket-name`.
    3.  Run `terraform plan` to see if your code matches the real settings. Adjust code until `plan` shows "No changes".

### Q5: You need to create 100 EC2 instances, but AWS limits you to 20. Terraform fails halfway. What happens to the state?
**Answer:**
*   **Partial State**: Terraform saves the state for the 20 created instances.
*   **Next Run**: When you fix the quota and run `apply` again, Terraform knows about the 20 existing ones and only tries to create the remaining 80. It is **Idempotent**.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use Ansible instead of Terraform for provisioning AWS resources. Do you agree?
**Answer:**
*   **Distinction**:
    *   **Terraform**: Infrastructure Provisioning (Creating VPCs, Servers, DBs). Best at Lifecycle management.
    *   **Ansible**: Configuration Management (Installing Nginx, patching OS inside the server).
*   **Recommendation**: Use **Terraform** to create the EC2 instance, then use **Ansible** (or User Data) to configure the software inside it. They are complementary.
