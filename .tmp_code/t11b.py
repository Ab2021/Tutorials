def first_year_cost(setup_hours, run_hours, infra_month, staff_rate, licence_fee=0):
    staff = (setup_hours + run_hours) * staff_rate
    infra = infra_month * 12
    return staff + infra + licence_fee

stacks = {
    "DuckDB + Python": (40, 120, 200, 85, 0),
    "Postgres + dbt + Airflow": (160, 200, 900, 85, 0),
    "Spark + Delta": (240, 300, 2500, 85, 0),
}
print(f"{'Stack':<28}{'Staff':>10}{'Infra':>10}{'Total':>10}")
for name, args in stacks.items():
    total = first_year_cost(*args)
    staff = (args[0] + args[1]) * args[3]
    infra = args[2] * 12
    print(f"{name:<28}{staff:>10,}{infra:>10,}{total:>10,}")
print()
print("Setup time is a one-off; run time and infrastructure recur.")
