import math

a, b, c, d = 150, 850, 300, 98_700
prr = (a/(a+b)) / (c/(c+d))
ror = (a*d) / (b*c)

se_ln_prr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
se_ln_ror = math.sqrt(1/a + 1/b + 1/c + 1/d)

prr_lo, prr_hi = prr*math.exp(-1.96*se_ln_prr), prr*math.exp(1.96*se_ln_prr)
ror_lo, ror_hi = ror*math.exp(-1.96*se_ln_ror), ror*math.exp(1.96*se_ln_ror)

print(f"2x2: a={a}, b={b}, c={c}, d={d:,}")
print(f"PRR: {prr:.2f}  95% CI {prr_lo:.2f} to {prr_hi:.2f}")
print(f"ROR: {ror:.2f}  95% CI {ror_lo:.2f} to {ror_hi:.2f}")
print()
print("A lower bound above 1 is a signal, not a causal finding.")
print("The comparison uses all other drugs and events in the database, not a control group.")
