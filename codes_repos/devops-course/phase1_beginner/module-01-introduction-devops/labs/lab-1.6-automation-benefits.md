# Lab 1.6: Automation Benefits (Toil Reduction)

## 🎯 Objective

Experience the difference between manual operations ("Toil") and automated solutions. You will perform a repetitive task manually to feel the pain, then write a script to automate it, quantifying the time and error reduction.

## 📋 Prerequisites

-   Completed Lab 1.4 (Toolchain setup).
-   Basic Python or Bash knowledge.
-   A "Toil" mindset (initially).

## 🧰 Required Tools

-   **Python 3**
-   **Text Editor**
-   **Terminal**

## 📚 Background

### What is Toil?

Google's Site Reliability Engineering (SRE) book defines **Toil** as work that is:
1.  **Manual**: Requires human hands.
2.  **Repetitive**: Done over and over.
3.  **Automatable**: Could be done by a machine.
4.  **Tactical**: Short-term value only.
5.  **No Long-term Value**: Doesn't improve the service permanently.
6.  **O(n) with service growth**: Scales linearly with traffic/users.

**Why Automate?**
-   **Speed**: Machines are faster.
-   **Consistency**: Machines don't make typos.
-   **Morale**: Engineers hate repetitive work.
-   **Cost**: Engineering time is expensive.

---

## 🔨 Hands-On Implementation

### Part 1: The Manual Nightmare (The "Before") 😫

**Scenario:** You are a sysadmin. Your boss sends you a list of 50 new employees. You need to generate a "Welcome Email" file for each one, containing their username and a temporary password.

1.  **Setup:**
    Create a file `employees.csv` with the following content (copy-paste):
    ```csv
    John Doe,Engineering
    Jane Smith,Marketing
    Bob Jones,Sales
    Alice Wonderland,Engineering
    Charlie Brown,HR
    ... (Imagine 45 more lines) ...
    ```

2.  **The Manual Task:**
    For the first 5 employees, manually create a file named `welcome_<firstname>.txt`.
    Content format:
    ```text
    Hello <First Name> <Last Name>,
    Welcome to the <Department> department!
    Your username is: <firstname>.<lastname>
    Your password is: Welcome2025!
    ```

3.  **Execute:**
    -   Create `welcome_john.txt`... type content... save.
    -   Create `welcome_jane.txt`... type content... save.
    -   Create `welcome_bob.txt`... type content... save.
    -   *Stop after 3.*

4.  **Measure:**
    -   How long did it take per file? (~30 seconds?)
    -   Did you make any typos?
    -   How long would 1000 employees take? (30s * 1000 = 8.3 hours!)

### Part 2: The Automated Solution (The "After") 🤖

**Objective:** Write a script to process the CSV and generate files instantly.

1.  **Create the Script:**
    Create `onboard_employees.py`.

    ```python
    import csv
    import os
    import time

    def generate_username(first, last):
        return f"{first.lower()}.{last.lower()}"

    def create_welcome_file(first, last, dept):
        filename = f"welcome_{first.lower()}.txt"
        username = generate_username(first, last)
        
        content = f"""Hello {first} {last},
    Welcome to the {dept} department!
    Your username is: {username}
    Your password is: Welcome2025!
    """
        
        with open(filename, "w") as f:
            f.write(content)
        print(f"✅ Created {filename}")

    def main():
        start_time = time.time()
        
        if not os.path.exists("employees.csv"):
            print("❌ Error: employees.csv not found!")
            return

        count = 0
        with open("employees.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    full_name = row[0]
                    dept = row[1]
                    # Split name
                    parts = full_name.split(" ")
                    if len(parts) >= 2:
                        create_welcome_file(parts[0], parts[1], dept)
                        count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 Processed {count} employees in {duration:.4f} seconds.")
        print(f"⚡ Speedup: {(30 * count) / duration:.0f}x faster than manual!")

    if __name__ == "__main__":
        main()
    ```

2.  **Run the Script:**
    ```bash
    python3 onboard_employees.py
    ```

3.  **Analyze Results:**
    -   **Time:** < 0.1 seconds.
    -   **Accuracy:** 100% consistent.
    -   **Scalability:** 1000 employees would take ~0.5 seconds.

### Part 3: Infrastructure Automation (IaC Preview) 🏗️

**Scenario:** You need to create 3 directories (`dev`, `staging`, `prod`) and inside each, a `config` folder and a `logs` folder.

**Manual Way:**
`mkdir dev`, `cd dev`, `mkdir config`, `mkdir logs`, `cd ..`, `mkdir staging`... (Too many commands!)

**Automated Way (Bash Script):**
Create `setup_env.sh`:

```bash
#!/bin/bash

ENVIRONMENTS=("dev" "staging" "prod")
SUBDIRS=("config" "logs" "data")

echo "🚀 Setting up infrastructure..."

for env in "${ENVIRONMENTS[@]}"; do
    echo "  📂 Creating environment: $env"
    for sub in "${SUBDIRS[@]}"; do
        path="$env/$sub"
        mkdir -p "$path"
        touch "$path/.gitkeep" # Create a placeholder file
        echo "    └── Created $path"
    done
done

echo "✅ Done!"
```

**Run it:**
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## 🎯 Challenges

### Challenge 1: Error Handling (Difficulty: ⭐⭐)

**Scenario:** The CSV file might have bad data.
`John,Engineering` (Missing last name)
`Jane Smith` (Missing department)

**Task:**
Modify `onboard_employees.py` to:
1.  Log errors to `error.log` instead of crashing or printing to screen.
2.  Skip the bad lines but continue processing the good ones.
3.  Print a summary at the end: "Processed X, Failed Y".

### Challenge 2: Idempotency (Difficulty: ⭐⭐⭐)

**Definition:** An operation is **idempotent** if running it multiple times produces the same result as running it once (no side effects).

**Scenario:**
If you run `onboard_employees.py` twice, it overwrites the files. That's okay.
But imagine the script was "Create User in Database". Running it twice would crash ("User already exists").

**Task:**
Modify the script to check if the welcome file already exists.
-   If it exists, print "⚠️  Skipping {filename} (Already exists)".
-   If not, create it.

---

## 💡 Solution

<details>
<summary>Click to reveal Challenge 1 & 2 Solution</summary>

```python
import csv
import os
import time
import datetime

def log_error(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("error.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def main():
    start_time = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0

    if not os.path.exists("employees.csv"):
        print("❌ Error: employees.csv not found!")
        return

    with open("employees.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader, 1):
            try:
                # Validation
                if len(row) < 2:
                    raise ValueError(f"Row {row_num}: Missing columns {row}")
                
                full_name = row[0].strip()
                dept = row[1].strip()
                
                parts = full_name.split(" ")
                if len(parts) < 2:
                    raise ValueError(f"Row {row_num}: Invalid name format '{full_name}'")
                
                first, last = parts[0], parts[1]
                filename = f"welcome_{first.lower()}.txt"
                
                # Idempotency Check (Challenge 2)
                if os.path.exists(filename):
                    print(f"⚠️  Skipping {filename} (Already exists)")
                    skip_count += 1
                    continue

                # Create File
                generate_username(first, last) # Just to test function
                create_welcome_file(first, last, dept) # Assume this function exists from Part 2
                success_count += 1

            except Exception as e:
                # Error Handling (Challenge 1)
                print(f"❌ Failed processing row {row_num}")
                log_error(str(e))
                fail_count += 1
    
    print(f"\n📊 Summary: Success: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

if __name__ == "__main__":
    # Helper functions from Part 2 would be here
    pass 
```

</details>

---

## 🔑 Key Takeaways

1.  **ROI of Automation**: Spending 1 hour to automate a 5-minute task is worth it *if* you do that task daily.
    -   *XKCD Automation Chart rule: If you shave 5 seconds off a task you do 50 times a day, you can spend 5 days automating it and break even over a year.*
2.  **Consistency**: Scripts are self-documenting processes.
3.  **Idempotency**: Crucial for DevOps. Scripts should be safe to re-run.

---

## ⏭️ Next Steps

Automation frees up time. What do we do with that time? We improve!

Proceed to **Lab 1.7: Continuous Improvement (Kaizen)**.
