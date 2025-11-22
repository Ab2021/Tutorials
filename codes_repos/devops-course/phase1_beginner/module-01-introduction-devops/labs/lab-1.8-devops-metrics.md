# Lab 1.8: DevOps Metrics (DORA)

## 🎯 Objective

Understand and calculate the four key **DORA (DevOps Research and Assessment)** metrics. You will simulate a series of deployments and incidents, log the data, and calculate your team's "DevOps Performance Level" (Elite, High, Medium, Low).

## 📋 Prerequisites

-   Completed Lab 1.7.
-   Spreadsheet software (Excel, Google Sheets) or Python for calculation.

## 📚 Background

### The DORA Metrics

The DORA research team identified four metrics that statistically correlate with high-performing organizations.

1.  **Deployment Frequency (DF)**: How often do you deploy code to production?
    -   *Elite:* On-demand (multiple deploys per day).
    -   *Low:* Between once per month and once every 6 months.

2.  **Lead Time for Changes (LT)**: Time from "commit" to "running in production".
    -   *Elite:* Less than one hour.
    -   *Low:* Between one month and 6 months.

3.  **Change Failure Rate (CFR)**: Percentage of deployments that cause a failure in production.
    -   *Elite:* 0-15%.
    -   *Low:* 46-60%.

4.  **Mean Time to Restore (MTTR)**: How long it takes to restore service after a failure.
    -   *Elite:* Less than one hour.
    -   *Low:* Between one week and one month.

---

## 🔨 Hands-On Implementation

### Part 1: Generating Data 📉

**Scenario:** You are the Release Manager. You have a log of the last month's activities.

1.  **Create `deployment_log.csv`:**

    ```csv
    ID,Commit_Time,Deploy_Time,Status,Restore_Time
    1,2025-11-01 09:00,2025-11-01 10:00,Success,
    2,2025-11-02 09:00,2025-11-02 14:00,Success,
    3,2025-11-03 10:00,2025-11-03 11:00,Failure,2025-11-03 13:00
    4,2025-11-05 09:00,2025-11-05 09:30,Success,
    5,2025-11-08 14:00,2025-11-08 15:00,Success,
    6,2025-11-10 09:00,2025-11-10 10:00,Failure,2025-11-10 10:30
    7,2025-11-12 11:00,2025-11-12 11:45,Success,
    8,2025-11-15 09:00,2025-11-15 09:30,Success,
    9,2025-11-20 10:00,2025-11-20 16:00,Success,
    10,2025-11-25 09:00,2025-11-25 09:15,Success,
    ```

### Part 2: Calculating Metrics 🧮

You can do this manually or write a script. Let's write a Python script `calculate_dora.py`.

1.  **Create the Script:**

    ```python
    import csv
    from datetime import datetime

    def parse_time(t_str):
        if not t_str: return None
        return datetime.strptime(t_str, "%Y-%m-%d %H:%M")

    def main():
        deployments = []
        failures = []
        lead_times = []
        restore_times = []

        with open("deployment_log.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                deployments.append(row)
                
                # Lead Time Calculation
                commit = parse_time(row["Commit_Time"])
                deploy = parse_time(row["Deploy_Time"])
                lead_time_hours = (deploy - commit).total_seconds() / 3600
                lead_times.append(lead_time_hours)
                
                # Change Failure Rate & MTTR
                if row["Status"] == "Failure":
                    failures.append(row)
                    restore = parse_time(row["Restore_Time"])
                    mttr_hours = (restore - deploy).total_seconds() / 3600
                    restore_times.append(mttr_hours)

        # 1. Deployment Frequency
        total_deploys = len(deployments)
        # Assuming data is for 1 month (30 days)
        deploys_per_day = total_deploys / 30
        
        # 2. Lead Time for Changes (Average)
        avg_lead_time = sum(lead_times) / len(lead_times)
        
        # 3. Change Failure Rate
        cfr = (len(failures) / total_deploys) * 100
        
        # 4. Mean Time to Restore
        avg_mttr = sum(restore_times) / len(restore_times) if restore_times else 0

        print("📊 DORA Metrics Report")
        print("=======================")
        print(f"1. Deployment Frequency: {deploys_per_day:.2f} per day (Total: {total_deploys})")
        print(f"2. Avg Lead Time:        {avg_lead_time:.2f} hours")
        print(f"3. Change Failure Rate:  {cfr:.1f}%")
        print(f"4. Mean Time to Restore: {avg_mttr:.2f} hours")
        
        print("\n🏆 Assessment:")
        assess_performance(deploys_per_day, avg_lead_time, cfr, avg_mttr)

    def assess_performance(df, lt, cfr, mttr):
        # Simplified Logic based on 2023 DORA report
        score = 0
        
        # DF: Elite > 1/day
        if df >= 1: score += 1
        
        # LT: Elite < 1 hour
        if lt < 1: score += 1
        
        # CFR: Elite < 5%
        if cfr < 5: score += 1
        
        # MTTR: Elite < 1 hour
        if mttr < 1: score += 1
        
        if score == 4: print("Level: Elite 🌟")
        elif score >= 2: print("Level: High 🚀")
        elif score >= 1: print("Level: Medium 🚗")
        else: print("Level: Low 🐢")

    if __name__ == "__main__":
        main()
    ```

2.  **Run the Script:**
    ```bash
    python3 calculate_dora.py
    ```

3.  **Analyze Output:**
    -   **DF**: 0.33/day (Low)
    -   **Lead Time**: ~2.5 hours (High/Elite)
    -   **CFR**: 20% (Medium/Low)
    -   **MTTR**: 1.25 hours (High)

### Part 3: Improving the Metrics 📈

**Scenario:** Management wants to reach "High" performance.

1.  **Improve Lead Time**:
    -   Look at Deployment ID 2 (5 hours) and ID 9 (6 hours).
    -   *Action:* Automate the build process (Lab 1.6).
    -   *Simulation:* Change those deploy times in CSV to be 30 mins after commit.

2.  **Improve CFR**:
    -   Look at ID 3 and 6.
    -   *Action:* Add automated tests (Lab 1.3).
    -   *Simulation:* Change ID 3 status to "Success" (caught by tests before deploy).

3.  **Re-run Script:**
    See how your score changes.

---

## 🎯 Challenges

### Challenge 1: Dashboarding (Difficulty: ⭐⭐)

**Task:**
If you have Excel or Google Sheets:
1.  Import the CSV.
2.  Create a chart showing "Lead Time" over time.
3.  Is it trending up (getting worse) or down (getting better)?

### Challenge 2: The "Hidden" Metric (Difficulty: ⭐⭐⭐)

**Background:** DORA metrics are great, but they don't measure **Operational Cost** or **User Satisfaction**.

**Task:**
Add a column to the CSV: `User_Happiness` (1-10).
Modify the script to calculate the correlation between `Deploy_Time` and `User_Happiness`.
*Hypothesis:* Does deploying faster make users happier? Or does it introduce bugs that make them sad?

---

## 🔑 Key Takeaways

1.  **You Get What You Measure**: If you only measure speed (DF), stability (CFR) might suffer. DORA balances speed vs. stability.
2.  **Trends Matter**: A single day's data is noise. Look at the trend over months.
3.  **Context**: "Low" performance isn't always bad. A legacy banking core might deploy once a month safely, and that's fine for them.

---

## ⏭️ Next Steps

We have the theory, the tools, and the metrics. Let's look at real-world examples.

Proceed to **Lab 1.9: DevOps Case Studies**.
