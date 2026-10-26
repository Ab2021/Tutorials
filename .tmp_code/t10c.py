exposed = 2_000_000
reports = 150
rate_per_100k = reports / exposed * 100_000
print(f"Exposed: {exposed:,}")
print(f"Reports: {reports}")
print(f"Crude reporting rate: {rate_per_100k:.2f} per 100,000 exposed")
print()
for under in (0.5, 0.8, 0.9, 0.95):
    true_rate = rate_per_100k / (1 - under)
    print(f"Under-reporting {under:.0%}: estimated true rate {true_rate:.1f} per 100,000")
print()
print("The reporting rate is a lower bound whose distance from the true rate")
print("depends on reporting completeness, which the database does not measure.")
