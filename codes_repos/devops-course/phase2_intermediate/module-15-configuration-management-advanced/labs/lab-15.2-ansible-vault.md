# Lab 15.2: Ansible Vault

## 🎯 Objective

Keep secrets secret. You should never commit passwords, API keys, or SSL certificates in plain text. **Ansible Vault** encrypts files so you can commit them to Git safely.

## 📋 Prerequisites

-   Ansible installed.

## 📚 Background

### How it works
-   You create a file `secrets.yml`.
-   You encrypt it with a password.
-   It looks like `AES256...` garbage text.
-   Ansible decrypts it in memory when running the playbook.

---

## 🔨 Hands-On Implementation

### Part 1: Create Encrypted File 🔐

1.  **Create `secrets.yml`:**
    ```bash
    ansible-vault create secrets.yml
    ```
    *Prompt:* Enter New Vault Password (e.g., `mypassword`).
    *Editor opens (vi/nano).*

2.  **Add Content:**
    ```yaml
    db_password: SuperSecretPassword123!
    api_key: ABC-DEF-GHI
    ```
    Save and exit.

3.  **View File:**
    ```bash
    cat secrets.yml
    ```
    *Result:* `$ANSIBLE_VAULT;1.1;AES256...` (Encrypted).

### Part 2: Edit Encrypted File ✏️

1.  **Edit:**
    ```bash
    ansible-vault edit secrets.yml
    ```
    Enter password.
    Change `db_password` to `NewPassword456`.
    Save.

### Part 3: Use in Playbook 🎭

1.  **Create `site.yml`:**
    ```yaml
    ---
    - hosts: localhost
      connection: local
      vars_files:
        - secrets.yml
      tasks:
        - name: Debug Secret (Don't do this in prod!)
          debug:
            msg: "The password is {{ db_password }}"
    ```

2.  **Run:**
    ```bash
    ansible-playbook site.yml --ask-vault-pass
    ```
    Enter password.
    *Result:* `The password is NewPassword456`.

### Part 4: Password File (Automation) 🤖

Typing the password every time is annoying for CI/CD.

1.  **Create `.vault_pass`:**
    ```bash
    echo "mypassword" > .vault_pass
    ```
    **IMPORTANT**: Add `.vault_pass` to `.gitignore`!

2.  **Run without prompt:**
    ```bash
    ansible-playbook site.yml --vault-password-file .vault_pass
    ```

---

## 🎯 Challenges

### Challenge 1: Encrypt String (Difficulty: ⭐⭐)

**Task:**
Instead of encrypting the whole file, encrypt just one variable string.
`ansible-vault encrypt_string 'SuperSecret' --name 'my_secret'`
Copy the output into a regular `vars.yml` file.
*Benefit:* You can see the variable name, just not the value.

### Challenge 2: Rekey (Difficulty: ⭐)

**Task:**
Change the vault password.
`ansible-vault rekey secrets.yml`
Enter old password, then new password.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Output looks like:
```yaml
my_secret: !vault |
          $ANSIBLE_VAULT;1.1;AES256
          ...
```
</details>

---

## 🔑 Key Takeaways

1.  **Git Safety**: Vault allows you to keep infrastructure public but secrets private.
2.  **CI/CD**: In Jenkins/GitHub Actions, inject the `.vault_pass` file as a Secret File.
3.  **Multiple Vaults**: You can have different passwords for Dev and Prod (`--vault-id dev@prompt --vault-id prod@prompt`).

---

## ⏭️ Next Steps

We have secured our config. Now let's look at Observability.

Proceed to **Module 16: Monitoring & Observability**.
