from pathlib import Path
p = Path(r"D:\Biomedical\biomedical-data-science-2026\modules\14A_hardware\07_power-and-thermal.md")
s = p.read_text(encoding="utf-8")

old = "The thermal path is a system design, not a component choice. A heatsink inside a sealed plastic case does little; the heat must reach the outside."
new = old + """

### 5.16 Power measurement techniques

| Method | Measures | Strength | Limitation |
|---|---|---|---|
| Multimeter | Average current | Simple | Hides pulses; burden voltage |
| Current probe and oscilloscope | Waveform and charge | Shows pulses | Requires access; bandwidth |
| Power analyzer | Power over time | Accurate; integrates | Cost; setup |
| Battery simulator | Current and voltage | Controlled; repeatable | Does not capture battery behaviour |
| Fuel gauge | State of charge | In-system | Requires calibration |
| Thermal measurement | Power via temperature | Non-invasive | Slow; indirect |

The method must match the question. A multimeter is adequate for a continuous load; a current probe is needed for a pulsed radio; a power analyzer is needed for a full profile. The measurement's bandwidth and integration time must be stated.

A common error is to measure the current with a multimeter in series and neglect the burden voltage. A 10 mA current through a 100 Ω shunt drops 1 V, which may reset the device or change its mode. Use a low-burden meter or a current probe.

### 5.17 Regulator efficiency and dropout

A linear regulator's efficiency is the output voltage divided by the input voltage. A 3.0 V output from a 4.2 V battery has an efficiency of 71%; the remaining 29% becomes heat. A switching regulator's efficiency is 85–95% at moderate load, but falls at light load because of switching losses and quiescent current.

| Regulator | Efficiency at 1 mA | Efficiency at 100 mA | Quiescent current |
|---|---:|---:|---:|
| Linear | 71% | 71% | 50 µA |
| Switching, PWM | 70% | 90% | 100 µA |
| Switching, PFM | 85% | 80% | 10 µA |
| Switching, auto | 85% | 90% | 20 µA |

A wearable that spends most of its time at 1 mA benefits from a low-quiescent, PFM-capable regulator. A device that runs at 100 mA benefits from a PWM regulator. The choice depends on the load profile, not on the peak efficiency.

The dropout voltage is the minimum input-output difference. A 3.3 V regulator with a 3.0 V dropout cannot maintain 3.3 V from a 4.2 V battery once the battery falls below 6.3 V; the device must stop before the battery is empty, which reduces the usable capacity.

### 5.18 Battery charging profiles

A lithium-ion charger uses constant current until the cell reaches 4.2 V, then constant voltage until the current falls below a threshold. The profile must match the chemistry; a 4.35 V charger on a 4.2 V cell shortens the life and increases the risk.

| Phase | Control | Termination | Risk if wrong |
|---|---|---|---|
| Precharge | Low current | Cell above 3.0 V | Damage to a deeply discharged cell |
| Constant current | Fixed current | Cell reaches 4.2 V | Overheating at high current |
| Constant voltage | Fixed voltage | Current below 0.05 C | Overcharge if voltage is high |
| Top-off | Optional | Timer | Unnecessary heat |
| Maintenance | Optional | Periodic | Overcharge |

The charger must also monitor the cell temperature and stop if it is outside the safe range. A cell charged below 0 °C can plate lithium, which is a safety hazard.

### 5.19 Thermal test methods

| Method | Measures | Strength | Limitation |
|---|---|---|---|
| Thermocouple | Point temperature | Accurate; cheap | Contact error; single point |
| Infrared camera | Surface map | Fast; non-contact | Emissivity; reflective surfaces |
| Thermal test die | Junction temperature | Direct | Requires a special part |
| Calorimetry | Heat generated | Accurate | Slow; laboratory |
| Accelerated life | Failure at temperature | Predicts life | Model-dependent |

The test must use the production enclosure, the production power mode, and the worst-case ambient. A test on an open board underestimates the case temperature because the convection is better and there is no enclosure.

The applied-part temperature must be measured at the skin contact surface, with the device worn or pressed against a representative surface. A thermocouple on the outside of the case may read lower than the skin interface because of the contact resistance.

### 5.20 The worst-case load profile

The worst-case profile is not the average profile. It is the combination of the highest duty cycles, the highest ambient, and the lowest battery voltage that the device must support.

| Parameter | Nominal | Worst case | Effect |
|---|---|---|---|
| Display duty | 10% | 100% | 10× display energy |
| Radio duty | 1% | 10% | 10× radio energy |
| Ambient | 22 °C | 40 °C | Lower battery capacity; higher temperature |
| Battery voltage | 3.8 V | 3.3 V | Higher current for the same power |
| Age | New | 2 years | 20% capacity loss |

A device that meets its life claim at nominal and fails at worst case has a claim that is true only for some users. The claim must state the conditions.

### 5.21 Ten power and thermal review questions

1. What is the load table, and which loads are measured?
2. What is the duty cycle in the intended use?
3. What is the battery derating, and how was it derived?
4. What is the regulator efficiency at the actual load?
5. What is the quiescent current of the whole system in sleep?
6. What is the peak current, and can the battery supply it?
7. What is the thermal resistance from the hot component to the ambient?
8. What is the thermal time constant, and how long is the test?
9. What is the applied-part temperature at the worst-case ambient?
10. What is recorded so that a battery or firmware change can be detected?

A review that answers all ten has covered the power and thermal system. A review that answers only the first and the third has reviewed the battery, not the device."""
assert old in s
s = s.replace(old, new, 1)

old = "The general method is to build the load table from measurements, apply the derating, compute the thermal rise and time constant, and test for at least three time constants. A power budget that omits a load is a budget that will be corrected by the field."
new = """The general method is to build the load table from measurements, apply the derating, compute the thermal rise and time constant, and test for at least three time constants. A power budget that omits a load is a budget that will be corrected by the field.

### 6.8 A second design: the 2 W edge device

An edge-AI handheld dissipates 2 W in a 100 mm × 80 mm × 20 mm plastic case. The surface area is:

\\[
A = 2(100 \\times 80 + 100 \\times 20 + 80 \\times 20)\\ \\text{mm}^2 = 0.0232\\ \\text{m}^2
\\]

With natural convection at 8 W/m²K, the rise is:

\\[
\\Delta T = \\frac{2}{8 \\times 0.0232} = 10.8\\ \\text{K}
\\]

At a 30 °C ambient, the case is 40.8 °C, just below the 41 °C disclosure threshold. At a 35 °C ambient, the case is 45.8 °C, above the 43 °C limit for prolonged contact. The device is safe on a desk and unsafe against skin.

The design response is to keep the hot surface away from the patient, to add a thermal break, or to reduce the power. A metal case would spread the heat but also conduct it to the skin; the safety analysis must consider the contact area and duration.

### 6.9 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Average power | Nominal profile, 25 °C | mW | < 3.0 mW |
| Average power | Worst-case profile, 40 °C | mW | < 4.5 mW |
| Battery life | Nominal profile, 25 °C | Days | > 5 days |
| Battery life | Worst-case, 2 years aged | Days | > 3 days |
| Peak current | Radio transmit | mA | < battery limit |
| Case temperature | 40 °C ambient, 60 min | °C | < 43 °C |
| Applied-part temperature | 40 °C ambient, 60 min | °C | < 43 °C; disclose > 41 °C |
| Thermal time constant | Step power | Minutes | < 15 min |
| Charger | 0–45 °C ambient | Profile | No overcharge; temperature stop |
| Brown-out | Battery at cut-off | Reset | Safe state |

The matrix tests the nominal and worst cases. A device that passes only the nominal is not verified for the intended use.

### 6.10 Battery life across temperatures

| Temperature | Usable capacity | Average power | Life |
|---|---:|---:|---:|
| 15 °C | 1,210 J | 3.2 mW | 4.4 days |
| 25 °C | 1,274 J | 2.93 mW | 5.0 days |
| 40 °C | 1,180 J | 2.93 mW | 4.7 days |

The capacity falls at low temperature because the cell's internal resistance rises; the average power rises because the display and the heater (if any) draw more. The life at 15 °C is 12% lower than at 25 °C. A claim of "5 days" without a temperature is incomplete.

The table also shows that the 7-day requirement is not met at any temperature. The design must change, not the test."""
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("ok")
