def total_cost(fee, compute_months, compute_per_month, staff_hours, staff_rate, legal_hours, legal_rate):
    compute = compute_months * compute_per_month
    staff = staff_hours * staff_rate
    legal = legal_hours * legal_rate
    return {
        "licence fee": fee,
        "compute": compute,
        "staff": staff,
        "legal": legal,
        "total": fee + compute + staff + legal,
    }

scenarios = {
    "Open academic dataset": (0, 3, 500, 120, 85, 8, 250),
    "Credentialed research DUA": (0, 6, 700, 180, 85, 30, 250),
    "Non-commercial RUA with paid commercial licence": (70000, 8, 900, 220, 85, 80, 300),
    "Licensed commercial EHR": (150000, 12, 1200, 300, 85, 120, 300),
}

print(f"{'Scenario':<50}{'Fee':>10}{'Compute':>10}{'Staff':>10}{'Legal':>10}{'Total':>12}")
for name, args in scenarios.items():
    c = total_cost(*args)
    print(f"{name:<50}{c['licence fee']:>10,}{c['compute']:>10,}{c['staff']:>10,}{c['legal']:>10,}{c['total']:>12,}")
