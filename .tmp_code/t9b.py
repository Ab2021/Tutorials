import math

p = 0.3909
n = 5000
deff = 2.5

se_srs = math.sqrt(p * (1 - p) / n)
se_complex = math.sqrt(deff * p * (1 - p) / n)
n_eff = n / deff

print(f"Prevalence: {p:.4f}")
print(f"Sample size: {n:,}")
print(f"Design effect: {deff}")
print(f"SRS standard error:      {se_srs:.5f} ({se_srs:.2%})")
print(f"Complex standard error:  {se_complex:.5f} ({se_complex:.2%})")
print(f"Effective sample size:   {n_eff:,.0f}")
print()
print(f"SRS 95% CI:     {p-1.96*se_srs:.4f} to {p+1.96*se_srs:.4f}")
print(f"Complex 95% CI: {p-1.96*se_complex:.4f} to {p+1.96*se_complex:.4f}")
print()
print(f"CI width ratio: {(2*1.96*se_complex)/(2*1.96*se_srs):.2f}")
