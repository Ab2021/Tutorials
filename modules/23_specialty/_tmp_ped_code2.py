import math

def bsa(height_cm, weight_kg):
    return math.sqrt(height_cm * weight_kg / 3600)

def dose_check(weight_kg, mg_per_kg, max_mg, plausible):
    if not plausible[0] <= weight_kg <= plausible[1]:
        return "REJECT", f"weight {weight_kg} kg outside {plausible[0]}-{plausible[1]} kg"
    dose = weight_kg * mg_per_kg
    if dose > max_mg:
        return "REJECT", f"{dose:.1f} mg exceeds cap {max_mg} mg"
    return "ACCEPT", f"{dose:.1f} mg"

print(f"BSA at 2.04 kg: {bsa(44, 2.04):.3f} m2")
print(f"BSA at 20.4 kg: {bsa(44, 20.4):.3f} m2")
for w in (2.04, 20.4, 2040):
    status, msg = dose_check(w, 5, 20.4, (0.5, 10.0))
    print(f"{w} kg -> {status}: {msg}")
status, msg = dose_check(2.04, 50, 20.4, (0.5, 10.0))
print(f"50 mg/kg order -> {status}: {msg}")
print(f"BSA dose at 100 mg/m2: {100 * bsa(44, 2.04):.1f} mg")
print(f"BSA dose at wrong weight: {100 * bsa(44, 20.4):.1f} mg")
