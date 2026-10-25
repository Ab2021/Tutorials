# Power and Thermal

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 07 of 14
> Estimated study time: 4 hours
> Prerequisites: 05, 06
> Version 2026.3

## 1. Why this topic exists

Every medical device has a power budget and a thermal budget, and both are clinical constraints. A wearable that lasts 18 hours instead of 7 days is a product failure. A handheld probe that reaches 45 °C is a safety issue. An implantable device that cannot deliver its therapy for five years requires surgery to replace. Power and thermal are not back-office engineering; they determine whether the device can be used.

The industry problem is that power is estimated from a component list and thermal is estimated from a datasheet, and both estimates are usually optimistic. A microcontroller's datasheet current excludes the peripherals that are actually enabled. A radio's average current depends on the connection interval, the payload, and the retry rate. A battery's capacity depends on the discharge rate, the temperature, and the age. The gap between the estimate and the field is where battery life claims fail.

Three decisions depend on power and thermal literacy:

1. **Whether the battery life claim is achievable.** The claim must be computed over the duty cycle, with the actual peripherals, at the actual temperature, with the actual battery derating.
2. **Whether the thermal design is safe.** The applied-part temperature limit is a regulatory requirement, and the patient's tolerance is lower than the limit for some groups.
3. **Whether the device can sustain its performance.** Thermal throttling can turn a fast device into a slow one after minutes of operation, invalidating a latency claim that was measured at startup.

This topic covers power budgeting, battery selection and derating, thermal resistance, self-heating, thermal time constants, and the standards that constrain temperature and batteries. It assumes the processor and SoC material from Topics 05 and 06.

A useful framing: energy is the currency, power is the rate, and heat is the tax. A device that uses energy efficiently but dissipates it in a small area can still overheat. A device that runs cool but drains the battery in a day is still a failure.

## 2. Learning outcomes

By the end of this topic you can:

- Build a power budget from a duty-cycle table and compute battery life with derating.
- Compute the thermal rise of an enclosure from its power, surface area, and heat-transfer coefficient.
- Estimate the thermal time constant and decide whether a transient or a steady-state analysis is needed.
- Apply the IEC 60601-1 temperature limits and explain the difference between the 41 °C disclosure threshold and the 43 °C limit.
- Choose a battery chemistry and a protection scheme from the energy, power, and safety requirements.
- Explain why a battery's usable capacity is lower than its nameplate capacity.
- Identify the thermal failure modes that change latency, accuracy, or safety.
- Write a power and thermal verification plan for a wearable or handheld device.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

Power flows from the battery through regulators to the loads, and the losses become heat. The heat flows from the components through the enclosure to the environment. The device's temperature is the ambient plus the rise, and the rise is the power times the thermal resistance.

```text
   BATTERY ---> REGULATOR ---> LOADS ---> HEAT ---> ENCLOSURE ---> AMBIENT
      |             |             |          |           |            |
   capacity       efficiency    duty       power      thermal      ambient
   derating       loss          cycle      map        resistance   temperature
      |             |             |          |           |            |
      +-------------+-------------+----------+-----------+------------+
                                  |
                            ENERGY BUDGET
```

The model has four equations:

\[
P_{\text{avg}} = \sum_i P_i \times d_i
\]

\[
t_{\text{battery}} = \frac{C \times V \times \eta \times \text{derating}}{P_{\text{avg}}}
\]

\[
\Delta T = P \times R_{\text{th}} = \frac{P}{h A}
\]

\[
\tau_{\text{th}} = R_{\text{th}} C_{\text{th}}
\]

The first gives the average power from the duty cycle; the second gives the battery life; the third gives the steady-state temperature rise; the fourth gives the time to reach it. The transient temperature is:

\[
T(t) = T_{\text{amb}} + \Delta T \left(1 - e^{-t/\tau}\right)
\]

A device that runs for less than three time constants does not reach steady state, and a steady-state thermal test overestimates its temperature. A device that runs for more than three time constants is at steady state, and a short test underestimates it.

### 3.1 The load map

A power budget is a table of loads, each with an active power, a duty cycle, and a sleep power.

| Load | Active power | Duty | Sleep power | Average |
|---|---:|---:|---:|---:|
| AFE | 0.51 mW | 100% | 0.05 mW | 0.51 mW |
| MCU | 5.0 mW | 5% | 0.05 mW | 0.30 mW |
| Radio | 10 mW | 1% | 0.001 mW | 0.10 mW |
| Flash | 1.0 mW | 2% | 0.001 mW | 0.02 mW |
| Display | 20 mW | 10% | 0.01 mW | 2.0 mW |
| **Total** | | | | **2.93 mW** |

The table is the foundation of the battery estimate. A load that is omitted is a load that extends the estimate beyond reality. The most commonly omitted loads are the display, the radio in connected mode, the sensor's bias current, and the regulator's quiescent current.

### 3.2 Battery derating

A battery's nameplate capacity is measured at a specific discharge rate, temperature, and cut-off voltage. The usable capacity is lower because:

- The discharge rate is higher than the rated rate (Peukert effect).
- The temperature is lower than 25 °C.
- The cut-off voltage is higher than the rated cut-off.
- The battery ages.
- The regulator has a minimum input voltage.
- The pulse current causes a voltage drop.

A typical derating for a wearable at 0–40 °C and a 2-year life is 70–80% of the nameplate. A design that uses 100% is a design that will fall short.

### 3.3 Thermal resistance

The thermal resistance from the device to the ambient has three components: the junction-to-case resistance, the case-to-ambient resistance, and the contact resistance. For a small enclosure, the case-to-ambient resistance dominates, and it depends on the surface area and the heat-transfer coefficient.

| Mechanism | Heat-transfer coefficient | Condition |
|---|---:|---|
| Natural convection | 5–10 W/m²K | Still air |
| Forced convection | 20–100 W/m²K | Fan or wind |
| Contact | 100–1,000 W/m²K | Metal to metal |
| Radiation | 5–10 W/m²K | Typical surfaces |

For a wearable against skin, the skin contact adds a parallel path with a lower resistance, which makes the skin warmer than the air. The applied-part temperature limit applies to the skin contact, not the air.

### 3.4 The temperature limits

IEC 60601-1 clause 11.1.2.2 sets the applied-part temperature requirements. An applied part not intended to supply heat that reaches above 41 °C must have its maximum temperature disclosed in the instructions for use. The absolute maximum for an applied part in contact with the patient for 10 minutes or more is 43 °C. The limits are lower for prolonged contact and for vulnerable patients, and the standard's rationale notes that infants and thermally high-risk groups may be harmed even at 43 °C.

The 41 °C and 43 °C figures are often confused. The 41 °C is a disclosure threshold; the 43 °C is the limit. A device at 42 °C is compliant if it discloses the temperature, but it is close to the limit and may be uncomfortable.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Energy | Capacity to do work | Confused with power | J |
| Power | Rate of energy use | Confused with energy | W |
| Duty cycle | Fraction of time active | Confused with probability | % |
| Average power | Power weighted by duty cycle | Confused with peak | W |
| Peak power | Maximum instantaneous power | Confused with average | W |
| Quiescent current | Current drawn when idle | Ignored in budget | µA |
| Battery capacity | Charge stored | Confused with energy | mAh |
| Nameplate capacity | Rated capacity at a specified rate | Assumed usable | mAh |
| Derating | Reduction from nameplate | Ignored | % |
| Cut-off voltage | Voltage at which the device stops | Confused with empty | V |
| Peukert effect | Capacity falls at higher discharge rates | Ignored | None |
| Internal resistance | Resistance of the cell | Causes voltage drop | mΩ |
| Self-discharge | Loss of charge over time | Ignored for long shelf life | %/month |
| Thermal resistance | Temperature rise per watt | Confused with conductivity | K/W |
| Thermal capacitance | Energy to raise temperature | Confused with resistance | J/K |
| Thermal time constant | Product of resistance and capacitance | Confused with settling | s |
| Heat-transfer coefficient | Heat flux per area per temperature | Confused with conductivity | W/m²K |
| Junction temperature | Temperature of the silicon | Confused with case | °C |
| Case temperature | Temperature of the package | Confused with ambient | °C |
| Ambient temperature | Temperature of the surrounding air | Confused with room | °C |
| Applied part | Part in contact with the patient | Confused with enclosure | Classification |
| Thermal throttling | Clock reduction at high temperature | Confused with failure | MHz, °C |
| Thermal runaway | Self-accelerating heating | Confused with overheating | None |
| State of charge | Fraction of capacity remaining | Confused with voltage | % |
| State of health | Capacity relative to new | Confused with charge | % |
| Coulombic efficiency | Charge out divided by charge in | Ignored | % |
| Regulator efficiency | Output power divided by input power | Ignored | % |
| Dropout voltage | Minimum input-output difference | Ignored | V |
| Leakage current | Current through insulation | Confused with quiescent | µA |
| Overcurrent protection | Circuit that limits current | Confused with fuse | A |
| Overvoltage protection | Circuit that limits voltage | Confused with clamp | V |
| Thermal pad | Conductive path to the case | Confused with heatsink | K/W |
| Heat pipe | Passive heat transporter | Overkill for small devices | K/W |

## 5. The mechanics, step by step

The procedure below builds a power and thermal budget and verifies it. Each step states its input, output, and rationale. The edge cases required by the assignment are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the operating profile

**Input:** the use case, the environment, and the clinical session.

**Output:** a timeline of modes, durations, and ambient conditions.

**Rationale:** the power budget depends on the duty cycle, and the duty cycle depends on the use case. A monitor that runs continuously has a different profile from a device that wakes every minute. A device used indoors at 22 °C has a different thermal profile from one used outdoors at 40 °C.

Write the profile as a table: mode, duration, repetition, ambient. A profile that says "typical use" is not a profile.

### Step 2. List the loads

**Input:** the schematic, the component datasheets, and the firmware.

**Output:** a table of loads with active power, sleep power, and duty cycle.

**Rationale:** every load must be listed, including the ones that are easy to forget: the display backlight, the radio in connected mode, the sensor bias, the reference, the regulator quiescent current, the LEDs, and the leakage through protection components.

A load that is omitted is a load that is not budgeted. The omission is usually the display or the radio, because they are intermittent and easy to overlook.

### Step 3. Measure the load currents

**Input:** a prototype and a current probe or a power analyzer.

**Output:** measured currents in each mode, with a tolerance.

**Rationale:** datasheet currents are typical values under specified conditions. The actual current depends on the firmware, the clock, the peripherals, and the temperature. Measure each mode with the production firmware, and record the distribution, not only the mean.

A measurement with a multimeter averages the pulses and hides the peak. A measurement with a current probe and an oscilloscope shows the pulse shape and the charge per event, which is what the battery sees.

### Step 4. Compute the average power

**Input:** the load table and the duty cycles.

**Output:** average power, peak power, and energy per day.

**Rationale:** the average power is the sum of the active and sleep powers weighted by the duty cycle. The peak power is the maximum simultaneous load, which determines the regulator and the battery's pulse capability.

The energy per day is the average power times 86,400 s. The battery life is the usable energy divided by the energy per day.

### Step 5. Choose the battery

**Input:** the energy per day, the peak power, the volume, the temperature, and the safety requirement.

**Output:** a chemistry, a capacity, a protection scheme, and a derating factor.

**Rationale:** the chemistry determines the energy density, the voltage, the temperature range, and the safety. Lithium-ion has a high energy density and a nominal 3.7 V; lithium-primary has a lower self-discharge and a 3.0 V nominal; nickel-metal-hydride is more tolerant of abuse but heavier.

The protection scheme must prevent overcharge, overdischarge, overcurrent, and short circuit. A device with a rechargeable battery needs a charger with the correct profile and a protection circuit. A device with a primary battery needs a fuse or a current-limiting circuit.

### Step 6. Apply the derating

**Input:** the nameplate capacity, the discharge rate, the temperature, the age, and the cut-off voltage.

**Output:** the usable capacity and the battery life.

**Rationale:** the derating is the difference between the nameplate and the usable energy. A 20% derating for temperature and age is typical; a 30% derating is conservative. The cut-off voltage matters: a device that stops at 3.3 V uses less of the cell's capacity than one that stops at 3.0 V.

The derating must be stated in the claim. A battery life of "7 days" with a 30% derating is a different claim from "7 days" at the nameplate.

### Step 7. Compute the thermal rise

**Input:** the average and peak power, the enclosure surface area, the heat-transfer coefficient, and the ambient.

**Output:** the steady-state temperature rise and the absolute temperature.

**Rationale:** the thermal rise is the power divided by the product of the heat-transfer coefficient and the surface area. For a small enclosure in still air, the coefficient is 5–10 W/m²K. For a device against skin, the contact path is in parallel and the local temperature is higher.

The calculation is a first-order estimate. A real enclosure has hot spots, internal air gaps, and contact resistances that the single-resistance model ignores. The estimate tells you whether the design is close to the limit; it does not replace a thermal test.

### Step 8. Compute the thermal time constant

**Input:** the thermal resistance and the thermal capacitance.

**Output:** the time constant and the time to steady state.

**Rationale:** the time constant is the product of the resistance and the capacitance. A small wearable has a time constant of minutes; a large device has a time constant of hours. A device that runs for less than three time constants does not reach steady state.

The time constant determines the test duration. A 30-minute test on a device with a 10-minute time constant reaches 95% of the steady-state rise; a 5-minute test reaches 39%.

### Step 9. Check the applied-part temperature

**Input:** the estimated or measured applied-part temperature and the contact duration.

**Output:** a compliance statement and a disclosure if required.

**Rationale:** IEC 60601-1 clause 11.1.2.2 requires disclosure above 41 °C and sets 43 °C as the absolute maximum for contact of 10 minutes or more. The measurement must be at the patient contact surface, under the worst-case ambient and workload, after thermal equilibrium.

A device that meets the limit at 22 °C may exceed it at 40 °C. The test must use the maximum ambient in the intended use environment.

### Step 10. Design the thermal path

**Input:** the hot components, the enclosure, and the available area.

**Output:** a thermal path: copper pour, thermal pad, heatsink, or case.

**Rationale:** the heat must travel from the junction to the case to the ambient. A thermal pad or a copper pour reduces the junction-to-case resistance; a metal case or a heatsink reduces the case-to-ambient resistance. An air gap is an insulator.

The thermal path must be designed with the mechanical and electrical constraints. A metal case can solve the thermal problem and create an electrical safety problem if it is not isolated.

### Step 11. Design the battery protection and charging

**Input:** the chemistry, the capacity, the charger, and the fault conditions.

**Output:** a protection circuit, a charger profile, and a fault response.

**Rationale:** a lithium battery can vent, catch fire, or explode if overcharged, overdischarged, or short-circuited. The protection circuit must be independent of the charger's control loop, and the charger must follow the chemistry's profile (constant current, then constant voltage).

The fault response must be safe. A device that continues to charge a faulty battery because the firmware does not detect the fault is a hazard.

### Step 12. Verify and document

**Input:** the prototype, the profile, and the limits.

**Output:** a power measurement, a thermal measurement, a battery life test, and a record.

**Rationale:** the estimates are hypotheses. Measure the average power over a full profile, the temperature at the applied part over a thermal soak, and the battery life over a discharge cycle at the worst-case temperature. Record the firmware, the battery lot, and the ambient.

The verification must include the worst case, not only the nominal. A device that passes at 22 °C and 50% duty may fail at 40 °C and 100% duty.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 4, 7 | Power exceeds the thermal limit | Temperature log | Throttling or shutdown |
| Common-mode rejection failure | 2 | Analogue issue | N/A | N/A |
| Electrode offset | 2 | Analogue issue | N/A | N/A |
| Clock jitter | 7, 9 | Thermal effect on oscillator | Frequency drift | Timing error |
| Brown-out | 4, 11 | Battery voltage drops under pulse load | Supply monitor | Reset; lost data |
| Watchdog reset loop | 12 | Firmware fails to service watchdog | Reset counter | Device unavailable |
| Thermal throttling | 7, 9 | Clock reduction at the limit | Temperature and latency | Reduced performance |
| Firmware update failure | 12 | Incomplete write; battery dies during update | Version and checksum | Bricked device |
| Calibration drift | 9, 12 | Temperature coefficient | Reference check | Bias |

The power and thermal edge cases are dominated by brown-out, thermal throttling, and battery failure. The analogue edge cases enter through the load map, not through the thermal model.

### 5.14 Battery chemistry comparison

| Chemistry | Nominal voltage | Energy density | Cycle life | Temperature range | Safety |
|---|---:|---:|---:|---|---|
| Li-ion (NMC) | 3.7 V | 150–250 Wh/kg | 500–1,000 | −20 to 60 °C | Requires protection |
| Li-polymer | 3.7 V | 150–250 Wh/kg | 500–1,000 | −20 to 60 °C | Requires protection; swelling |
| LiFePO4 | 3.2 V | 90–120 Wh/kg | 2,000–5,000 | −20 to 60 °C | More tolerant |
| Lithium primary | 3.0 V | 200–300 Wh/kg | Single use | −40 to 85 °C | High energy; transport rules |
| NiMH | 1.2 V | 60–120 Wh/kg | 500–1,000 | −20 to 50 °C | Tolerant of abuse |
| Alkaline | 1.5 V | 100–150 Wh/kg | Single use | −20 to 50 °C | Leakage risk |

The chemistry choice depends on the energy, the temperature, the cycle life, and the safety. An implantable device uses a lithium primary cell for its low self-discharge and high energy density; a wearable uses a Li-polymer for its shape flexibility; a backup device may use LiFePO4 for its tolerance.

### 5.15 Thermal path options

| Option | Thermal resistance | Cost | When to use |
|---|---:|---|---|
| Natural convection, plastic case | 20–40 K/W | Low | Low-power wearable |
| Natural convection, metal case | 10–20 K/W | Moderate | Handheld |
| Thermal pad to case | 5–15 K/W | Moderate | Hot component |
| Heatsink | 2–10 K/W | Moderate | Benchtop |
| Fan | 1–5 K/W | High | High-power |
| Heat pipe | 0.5–2 K/W | High | Compact high-power |

The thermal path is a system design, not a component choice. A heatsink inside a sealed plastic case does little; the heat must reach the outside.

## 6. Worked example

### 6.1 The requirement

A wearable cardiac monitor must run for 7 days on a single charge. The device is 40 mm × 30 mm × 10 mm, with a plastic case and a skin-contact electrode. The ambient range is 15–40 °C. The applied part must not exceed the IEC 60601-1 limits.

The load table is:

| Load | Active power | Duty | Sleep power | Average |
|---|---:|---:|---:|---:|
| AFE | 0.51 mW | 100% | 0.05 mW | 0.51 mW |
| MCU | 5.0 mW | 5% | 0.05 mW | 0.30 mW |
| Radio | 10.0 mW | 1% | 0.001 mW | 0.10 mW |
| Flash | 1.0 mW | 2% | 0.001 mW | 0.02 mW |
| Display | 20.0 mW | 10% | 0.01 mW | 2.00 mW |
| **Total** | | | | **2.93 mW** |

The battery is a 150 mAh Li-polymer at 3.7 V. The regulator efficiency is 85%. The derating for temperature and age is 25%.

### 6.2 The energy and battery life

The nameplate energy is:

\[
E_{\text{name}} = 0.150\ \text{Ah} \times 3.7\ \text{V} \times 3600 = 1{,}998\ \text{J}
\]

The usable energy is:

\[
E_{\text{usable}} = 1{,}998 \times 0.85 \times 0.75 = 1{,}274\ \text{J}
\]

The daily energy is:

\[
E_{\text{day}} = 2.93\ \text{mW} \times 86{,}400 = 253.2\ \text{J}
\]

The battery life is:

\[
t = \frac{1{,}274}{253.2} = 5.03\ \text{days}
\]

The device does not meet the 7-day requirement. The shortfall is 1.97 days, or 28%. The design must reduce the average power, increase the battery, or relax the requirement.

### 6.3 The thermal rise

The surface area of the enclosure is:

\[
A = 2(40 \times 30 + 40 \times 10 + 30 \times 10)\ \text{mm}^2 = 3{,}800\ \text{mm}^2 = 0.0038\ \text{m}^2
\]

With natural convection at 10 W/m²K and 2.93 mW, the rise is:

\[
\Delta T = \frac{0.00293}{10 \times 0.0038} = 0.077\ \text{K}
\]

The thermal rise is negligible. The device is not a thermal problem at 2.93 mW. At 40 °C ambient, the case is 40.08 °C, well below the 41 °C disclosure threshold and the 43 °C limit.

The thermal time constant requires the thermal capacitance. For a 20 g device with an effective specific heat of 1,000 J/kgK, the capacitance is 20 J/K. The time constant is:

\[
\tau = R_{\text{th}} C_{\text{th}} = \frac{1}{10 \times 0.0038} \times 20 = 526\ \text{s} = 8.8\ \text{min}
\]

The device reaches 95% of its steady-state rise in about 26 minutes. A 30-minute thermal test is adequate; a 5-minute test is not.

### 6.4 Diagnostic 1: the quiescent current is too high

The measured battery life is 3.1 days, not the predicted 5.03 days. The average current is 8.2 mA instead of 5.0 mA. The difference is 3.2 mA, which is far more than any load in the table.

The diagnostic is to measure the current in each mode with the device idle. The measurement shows 3.2 mA in "sleep", not the 0.05 mA assumed. The cause is the display's standby current and the regulator's quiescent current, neither of which was in the load table.

The design flaw is the omission of the quiescent loads. A regulator with 3 mA quiescent current, a display with 1 mA standby, and a sensor with 0.2 mA bias add 4.2 mA, which is 84% of the total. The fix is a low-quiescent regulator, a display that can be fully powered down, and a sensor with a shutdown mode.

### 6.5 Diagnostic 2: the thermal test is too short

A 10-minute thermal test at 40 °C shows a case temperature of 40.1 °C, which passes. A 60-minute test shows 40.4 °C, which still passes. The difference is small because the power is small. For a higher-power device, the difference would be larger.

The diagnostic is to compute the time constant before the test and to run for at least three time constants. A test shorter than one time constant reaches only 63% of the rise and can pass a device that would fail at equilibrium.

The design flaw is the test plan, not the device. The plan assumed that 10 minutes was enough without computing the time constant. The fix is to compute the time constant from the thermal resistance and capacitance, and to set the test duration to at least three time constants.

### 6.6 Vary one assumption: the display duty rises to 50%

Change only the display duty cycle, from 10% to 50%. The display's average power rises from 2.00 mW to 10.0 mW, and the total rises from 2.93 mW to 10.93 mW. The battery life becomes:

\[
t = \frac{1{,}274}{10.93 \times 10^{-3} \times 86{,}400} = 1.35\ \text{days}
\]

The device now lasts 1.35 days, far short of 7 days. The display is 91% of the average power. The thermal rise becomes 0.29 K, still negligible, but the battery life is a failure.

The consequence is that the display, not the AFE or the radio, is the dominant load. A design that focuses on the AFE's 0.51 mW while ignoring the display's 10 mW has optimised the wrong component.

### 6.7 What the example demonstrates

The arithmetic closes: 1,998 J nameplate, 1,274 J usable, 253 J per day, 5.03 days, 0.077 K rise, and an 8.8-minute time constant. The two diagnostics reveal different flaws: one is an omitted quiescent load, the other is a test that is too short. The sensitivity analysis shows that the display duty cycle, not the AFE, determines the battery life.

The general method is to build the load table from measurements, apply the derating, compute the thermal rise and time constant, and test for at least three time constants. A power budget that omits a load is a budget that will be corrected by the field.
