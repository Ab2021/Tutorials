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

### 3.5 Power and energy are different claims

A power claim and an energy claim answer different questions. "The device draws 3 mW" is a power claim; "the device runs for 5 days" is an energy claim. The first can be true while the second is false if the battery is small or the derating is large.

| Claim | Quantity | Depends on |
|---|---|---|
| "Low power" | Average power | Duty cycle, loads |
| "Long battery life" | Energy / average power | Capacity, derating, temperature |
| "Fast charge" | Charge rate | Charger, cell, thermal |
| "Cool operation" | Temperature rise | Power, area, heat transfer |

A design review should state which claim is being made and which quantity supports it. A "low-power" device with a 50 mAh battery may have a shorter life than a "high-power" device with a 500 mAh battery.

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

A review that answers all ten has covered the power and thermal system. A review that answers only the first and the third has reviewed the battery, not the device.

### 5.22 Energy harvesting

An energy-harvesting device replaces or supplements the battery with a source that converts ambient energy: light, motion, heat, or radio frequency. The source's power density is low, and the conversion is intermittent.

| Source | Power density | Condition | Challenge |
|---|---:|---|---|
| Photovoltaic | 10–100 µW/cm² | Indoor light | Low light; shadowing |
| Thermoelectric | 10–100 µW/cm² | 5–10 K gradient | Small gradient on skin |
| Piezoelectric | 10–100 µW/cm³ | Motion | Intermittent; high impedance |
| RF | 1–10 µW/cm² | Near a transmitter | Distance; regulation |

An energy-harvesting device needs a power-management circuit that can start from a low voltage, store energy, and deliver a pulse. The storage is often a supercapacitor or a thin-film battery. The design must handle the case where the source is absent for longer than the storage can support.

The data implication is that the device may brown out or reset when the source is low. The reset must be detected and the data must be marked, or the analysis will treat a gap as a physiological event.

### 5.23 Battery ageing

A battery's capacity falls and its internal resistance rises with age and cycling. The ageing is accelerated by high temperature, high state of charge, and high discharge rates.

| Stress | Effect | Mitigation |
|---|---|---|
| High temperature | Faster capacity loss | Thermal design; avoid charging hot |
| High state of charge | Faster calendar ageing | Store at 40–60% charge |
| Deep discharge | Capacity loss; safety risk | Cut-off voltage; protection |
| High charge rate | Lithium plating; heat | Charge profile; temperature monitor |
| High cycle count | Capacity fade | Larger battery; lower depth of discharge |

A device that claims a 2-year life must test the battery after accelerated ageing, not only when new. The accelerated test uses a higher temperature and a shorter time, with an Arrhenius or a power-law model to extrapolate. The model is an inference; the test is the evidence.

### 5.24 Thermal interface materials

A thermal interface material (TIM) fills the air gap between a hot component and a heatsink or case. Air is a poor conductor (about 0.026 W/mK); a TIM is 1–10 W/mK. The interface resistance is the thickness divided by the conductivity and the area.

| Material | Conductivity | Thickness | Use |
|---|---:|---:|---|
| Air | 0.026 W/mK | 0.1 mm | No interface |
| Gap pad | 1–5 W/mK | 0.5–2 mm | Rough surfaces |
| Thermal grease | 1–8 W/mK | 0.05 mm | Smooth surfaces |
| Phase-change | 1–5 W/mK | 0.05 mm | High power |
| Solder | 30–60 W/mK | 0.05 mm | High reliability |

A 0.5 mm gap pad at 3 W/mK over 1 cm² has a resistance of 1.67 K/W, which can dominate the thermal path. The interface is not a detail; it is often the largest resistance.

The interface material must be selected with the surface roughness and the pressure. A gap pad conforms to a rough surface but has a low conductivity; a grease has a high conductivity but can pump out over time. The interface is a reliability item, not only a thermal one.


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

The measured battery life is 3.1 days, not the predicted 5.02 days. The average current is 8.2 mA instead of 5.0 mA. The difference is 3.2 mA, which is far more than any load in the table.

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

The arithmetic closes: 1,998 J nameplate, 1,274 J usable, 254 J per day, 5.02 days, 0.077 K rise, and an 8.8-minute time constant. The two diagnostics reveal different flaws: one is an omitted quiescent load, the other is a test that is too short. The sensitivity analysis shows that the display duty cycle, not the AFE, determines the battery life.

The general method is to build the load table from measurements, apply the derating, compute the thermal rise and time constant, and test for at least three time constants. A power budget that omits a load is a budget that will be corrected by the field.

### 6.8 A second design: the 2 W edge device

An edge-AI handheld dissipates 2 W in a 100 mm × 80 mm × 20 mm plastic case. The surface area is:

\[
A = 2(100 \times 80 + 100 \times 20 + 80 \times 20)\ \text{mm}^2 = 0.0232\ \text{m}^2
\]

With natural convection at 8 W/m²K, the rise is:

\[
\Delta T = \frac{2}{8 \times 0.0232} = 10.8\ \text{K}
\]

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

The table also shows that the 7-day requirement is not met at any temperature. The design must change, not the test.

### 6.11 A 40 °C ambient with a 2 W load

If the same enclosure dissipates 2 W instead of 2.9 mW, the rise is:

\[
\Delta T = \frac{2}{10 \times 0.0038} = 52.6\ \text{K}
\]

At a 40 °C ambient, the case would be 92.6 °C, far above any safe limit. The enclosure cannot dissipate 2 W by natural convection. The design needs a metal case, a larger area, or a fan.

| Power | Rise at 10 W/m²K | Case at 40 °C | Feasible? |
|---:|---:|---:|---|
| 2.9 mW | 0.08 K | 40.1 °C | Yes |
| 0.5 W | 13.2 K | 53.2 °C | No for skin contact |
| 2 W | 52.6 K | 92.6 °C | No |
| 5 W | 131.6 K | 171.6 °C | No |

The table shows the scaling: a 690× increase in power produces a 690× increase in rise. The thermal design that works for a wearable does not work for an edge device. The architecture and the thermal design must be chosen together.

A thermal summary:

- The case temperature is the ambient plus the rise.
- The rise is the power divided by the heat-transfer coefficient times the area.
- The time constant is the resistance times the capacitance.
- The applied-part limit is 43 °C for contact of 10 minutes or more; 41 °C triggers disclosure.
- The test must use the production enclosure and the worst-case ambient.
- **Thermal via** — a copper path through a board.
- **Heat spreader** — a high-conductivity plate that distributes heat.
- **Thermal transient** — temperature change before steady state.
- **Worst-case profile** — the combination of highest load and ambient.
## 7. Code, query, or calculation

The script reproduces the load table, battery life, thermal rise, and thermal time constant in Section 6. It computes each load's average from its active power, duty cycle, and sleep power, then sums the results.

**Assumptions.** The load values from Section 6.1, a 150 mAh Li-polymer cell at 3.7 V, 85% regulator efficiency, 25% derating, a 40 mm × 30 mm × 10 mm enclosure, a 10 W/m²K heat-transfer coefficient, and a 20 J/K thermal capacitance.

```python
loads = [
    ("AFE", 0.51e-3, 1.00, 0.05e-3),
    ("MCU", 5.0e-3, 0.05, 0.05e-3),
    ("Radio", 10.0e-3, 0.01, 0.001e-3),
    ("Flash", 1.0e-3, 0.02, 0.001e-3),
    ("Display", 20.0e-3, 0.10, 0.01e-3),
]
avg = 0.0
print("LOAD TABLE")
for name, p_active, duty, p_sleep in loads:
    p_avg = p_active*duty + p_sleep*(1-duty)
    avg += p_avg
    print(f"  {name:8s} {p_avg*1e3:7.3f} mW")
print(f"  total    {avg*1e3:7.3f} mW")

cap_ah = 0.150
vbat = 3.7
eff = 0.85
derate = 0.75
energy_name = cap_ah*vbat*3600
energy_use = energy_name*eff*derate
day = avg*86400
life = energy_use/day

print()
print("ENERGY")
print(f"  nameplate       {energy_name:8.1f} J")
print(f"  usable          {energy_use:8.1f} J")
print(f"  per day         {day:8.1f} J")
print(f"  life            {life:8.2f} days")

A = 2*(40e-3*30e-3 + 40e-3*10e-3 + 30e-3*10e-3)
h = 10.0
dt = avg/(h*A)
Cth = 20.0
Rth = 1/(h*A)
tau = Rth*Cth
print()
print("THERMAL")
print(f"  area            {A*1e4:8.2f} cm2")
print(f"  rise            {dt:8.3f} K")
print(f"  Rth             {Rth:8.2f} K/W")
print(f"  time constant   {tau/60:8.2f} min")

print()
print("VARIANT: display duty 50%")
avg2 = avg - 20e-3*0.10 - 0.01e-3*0.90 + 20e-3*0.50 + 0.01e-3*0.50
print(f"  average         {avg2*1e3:8.2f} mW")
print(f"  life            {energy_use/(avg2*86400):8.2f} days")
```

**Expected output.**

```text
LOAD TABLE
  AFE        0.510 mW
  MCU        0.298 mW
  Radio      0.101 mW
  Flash      0.021 mW
  Display    2.009 mW
  total      2.938 mW

ENERGY
  nameplate         1998.0 J
  usable            1273.7 J
  per day            253.9 J
  life                5.02 days

THERMAL
  area               38.00 cm2
  rise               0.077 K
  Rth                26.32 K/W
  time constant       8.77 min

VARIANT: display duty 50%
  average            10.93 mW
  life                1.35 days
```

The output shows four things.

First, the display is 68% of the average power (2.009 mW of 2.938 mW). The AFE is 17%, the MCU 10%, the radio 3%, and the flash 1%. A design that focuses on the AFE's noise and ignores the display's duty cycle has optimised the smaller load.

Second, the battery life is 5.02 days, not the 7 days required. The shortfall is the derating (25%) plus the regulator efficiency (15%), which together reduce the usable energy from 1,998 J to 1,274 J. A claim based on the nameplate would be 7.9 days, which is 57% too high.

Third, the thermal rise is 0.077 K, which is negligible. The device is not thermally limited at 2.9 mW. The thermal design matters for the 2 W edge device, not for this wearable.

Fourth, the thermal time constant is 8.77 minutes, so a 30-minute test reaches steady state and a 5-minute test does not. The test duration follows from the time constant, not from a convention.

**A note on the derating.** The 25% derating is an assumption. A conservative design uses 30%; an aggressive one uses 20%. The choice must be justified by the battery's data and the use profile. A design that uses 0% is not a design; it is a hope.

### 7.1 Thermal transient

The steady-state rise is reached exponentially. The script prints the fraction of the steady-state rise at 5, 10, 20, and 30 minutes for a 526 s time constant.

```python
import math

tau = 526.0
for minutes in (5, 10, 20, 30):
    t = minutes * 60
    fraction = 1 - math.exp(-t / tau)
    print(f"{minutes:2d} min: {100*fraction:6.1f}% of steady state")
```

**Expected output.**

```text
 5 min:   43.5% of steady state
10 min:   68.1% of steady state
20 min:   89.8% of steady state
30 min:   96.8% of steady state
```

A 5-minute test sees less than half the rise. A 30-minute test sees 96.8%, which is close enough for most purposes. The test duration should be at least three time constants, or 26.3 minutes.

## 8. Data, standards, and regulatory context

Power and thermal behaviour is constrained by the general safety standard, the battery standard, the home-healthcare collateral standard, and the risk management standard.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard; binding when referenced | Applied-part temperature, enclosure temperature, leakage, protection |
| IEC 60601-1-11:2015+AMD1:2020 | Home healthcare environment | Collateral standard | Temperature, humidity, mechanical, and electrical requirements for home use |
| IEC 62133-2:2017+AMD1:2021 | Portable lithium battery safety | Consensus standard; FDA-recognized | Cell and battery safety tests, protection, marking |
| UN 38.3 | Transport of lithium batteries | Binding for transport | Vibration, shock, altitude, thermal, short circuit |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis for thermal and battery failure |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Battery management firmware, thermal protection |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Immunity of power supplies and battery chargers |

### 8.1 The temperature limits in IEC 60601-1

IEC 60601-1 clause 11.1.2.2 sets the applied-part temperature requirements. An applied part not intended to supply heat that reaches above 41 °C must have its maximum temperature disclosed in the instructions for use. The absolute maximum for an applied part in contact with the patient for 10 minutes or more is 43 °C. The standard's Annex A rationale notes that the third edition raised the limit from 41 °C to 43 °C but retained the disclosure requirement above 41 °C, and that infants and thermally high-risk groups may be harmed even at 43 °C.

Table 23 of the standard covers parts likely to be touched (the enclosure); Table 24 covers skin contact with applied parts. The limits depend on the material and the contact duration. A metal enclosure that is comfortable at 40 °C may be uncomfortable at 43 °C because metal conducts heat to the skin more efficiently than plastic.

The measurement must be at the patient contact surface, under the worst-case ambient and workload, after thermal equilibrium. A measurement on the bench at 22 °C does not demonstrate compliance at 40 °C.

### 8.2 The battery standard

IEC 62133-2:2017+AMD1:2021 covers the safety of portable sealed lithium cells and batteries. It includes tests for external short circuit, free fall, thermal abuse, crush, overcharge, forced discharge, and forced internal short circuit. The standard is a component standard; it does not evaluate the end product's charging, load, or environmental stress.

The FDA recognized IEC 62133-2 Edition 1.1 (2021-07) as a consensus standard for medical devices in 2025, with "Complete" extent. The recognition means that a declaration of conformity can support a premarket submission, not that the battery is safe in every device. The end-product standard, IEC 60601-1, still applies.

UN 38.3 governs transport. A lithium battery that is shipped must pass the UN tests, regardless of its medical use. The tests include altitude, thermal cycling, vibration, shock, external short circuit, impact, overcharge, and forced discharge.

### 8.3 Home healthcare

IEC 60601-1-11:2015+AMD1:2020 adds requirements for devices used in the home. The environment is harsher than a hospital: the ambient temperature can be higher, the power supply can be unreliable, and the device may be used by a lay person. The standard constrains the enclosure temperature, the battery, and the mechanical robustness.

A device that is compliant in a professional healthcare facility may not be compliant in the home. The intended-use environment determines the standard, and the data team should know which environment the device was validated for.

### 8.4 What the standards do not cover

The standards do not specify:

- The battery chemistry or capacity.
- The power budget or the duty cycle.
- The thermal design or the enclosure material.
- The charging profile, beyond the safety tests.
- The battery life claim.

Those are design decisions that the manufacturer must justify. A device can meet every clause and still last only half as long as claimed, because the claim was not tested at the worst-case temperature.

### 8.5 The data implications

The power and thermal state are part of the measurement. The battery voltage, the state of charge, the temperature, and the power mode must be recorded with the data. A change in any of them can change the sensor's behaviour, the processor's latency, and the device's availability.

A practical rule: if the dataset does not contain the battery state and the device temperature, the analysis cannot distinguish a physiological change from a power or thermal effect.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 60601-1 | Applied-part temperature, ambient, test duration | Demonstrate compliance |
| IEC 60601-1-11 | Home environment conditions | Demonstrate intended use |
| IEC 62133-2 | Cell type, protection, test report | Battery safety |
| UN 38.3 | Transport test report | Shipping |
| ISO 14971 | Thermal and battery hazard analysis | Link failure to harm |
| IEC 62304 | Charger and protection firmware version | Reproduce behaviour |

The record is the same information the data team needs for provenance. A team that records the battery state and temperature for its own analysis is also assembling regulatory evidence.

### 8.7 A note on transport and shelf life

A device with a lithium battery is regulated for transport. UN 38.3 requires tests for altitude, thermal cycling, vibration, shock, external short circuit, impact, overcharge, and forced discharge. A device that has not passed UN 38.3 cannot be shipped by air, regardless of its medical clearance.

Shelf life is a separate claim. A device stored for two years may have a battery that has self-discharged and aged. The shelf-life claim must state the storage temperature and the state of charge. A device stored at 40 °C ages faster than one stored at 20 °C.

### 9.6 What good battery practice looks like

- Use a cell with a documented safety file and a protection circuit.
- Measure the discharge curve at the intended temperature.
- Set the cut-off voltage from the device's minimum, not from the cell's minimum.
- Test the charger at the temperature extremes.
- Record the battery lot and the cycle count.
- Replace the battery before its capacity falls below the device's requirement.
- Design the enclosure so that a venting cell does not injure the patient.

## 9. Industry practice and case evidence

### 9.1 Battery life claims are usually conditional

A battery life claim is a statement about a profile. A wearable that claims "7 days" may mean 7 days at 25 °C, 10% display duty, and 1% radio duty. A user who reads the display more often, or uses the device in a cold environment, will see a shorter life.

The honest claim states the conditions: "up to 7 days at 25 °C with 10% display use and 1% radio transmission; 4 days at 15 °C with 50% display use." A claim without conditions is a marketing statement, not a specification.

**Documented fact:** the derating effects are standard battery behaviour. **Inference:** the specific life for a device depends on its profile, which must be measured.

### 9.2 Thermal throttling in wearables and handhelds

A wearable or handheld device that runs a processor at high utilization will heat up. The device may reduce the clock to stay within its thermal limit, which increases the latency and reduces the performance. The throttle point depends on the enclosure, the ambient, and the workload.

For a device that runs a model continuously, the sustained performance may be half the peak. A latency claim measured at startup is not a sustained claim. The verification must include a thermal soak.

**Documented fact:** thermal throttling is a standard behaviour of processors and SoCs. **Inference:** the throttle point for a specific device must be measured in its enclosure.

### 9.3 Implantable battery longevity

An implantable device's battery must last years, and replacing it requires surgery. The design uses a lithium primary cell with a low self-discharge rate, a known discharge profile, and a conservative end-of-life voltage. The battery's capacity is estimated from the therapy's energy per pulse, the pacing rate, and the quiescent current.

The estimation is a model, and the model is updated from the device's telemetry. A device that predicts 8 years and delivers 6 is a clinical problem, not only an engineering one. The manufacturer's longevity claim is part of the labelling.

**Documented fact:** implantable devices use primary lithium cells and publish longevity estimates. **Inference:** the actual longevity depends on the therapy and the patient, which vary.

### 9.4 Energy harvesting in wearables

Energy harvesting from light, motion, or body heat can extend a wearable's life, but the power density is low. A photovoltaic cell indoors may produce 10–100 µW/cm²; a thermoelectric generator on skin may produce 10–100 µW/cm² with a 5–10 K gradient. The harvested power is enough for a low-power sensor but not for a display or a radio.

The design must handle the intermittent source with a storage element and a power-management circuit. The device may need to buffer energy for a transmission burst, which means the average power is lower than the peak.

**Documented fact:** the power densities are representative values from the literature. **Inference:** the feasibility for a specific device depends on its average power and its storage.

### 9.5 What good power and thermal practice looks like

Teams that avoid power and thermal surprises tend to:

- Measure the current in every mode, with the production firmware.
- Include the quiescent loads, the display, and the radio.
- Apply a derating and state it.
- Compute the thermal time constant before the test.
- Test at the worst-case ambient and the worst-case workload.
- Measure the applied-part temperature at the contact surface.
- Record the battery lot, the firmware, and the power mode.
- Re-test after any hardware or firmware change.

The practices are ordinary. The failures they prevent are not.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Battery life shortfall | Measured current above the table | Omitted quiescent load | Device dies early | Low-quiescent parts; measure all modes |
| Battery derating ignored | Claim uses nameplate capacity | Temperature, rate, age | 20–30% shortfall | Apply and state derating |
| Brown-out under pulse | Reset during radio transmit | Battery internal resistance | Lost data | Bulk capacitor; battery with lower resistance |
| Thermal throttle | Latency rises after minutes | Clock reduction at the limit | Sustained latency miss | Thermal design; lower power |
| Applied part too hot | Case above 41 °C | Power dissipation; enclosure | Discomfort; non-compliance | Thermal break; lower power; disclosure |
| Charger overcharge | Cell voltage above 4.2 V | Wrong profile; failed control | Fire; venting | Independent protection; temperature stop |
| Deep discharge | Cell below 2.5 V | No cut-off | Capacity loss; safety | Protection circuit; cut-off |
| Thermal runaway | Rapid temperature rise | Internal short; abuse | Fire; injury | Cell qualification; protection; enclosure |
| Self-heating error | Sensor reads high | Excitation power | Bias in measurement | Low excitation; pulsed measurement |
| Regulator dropout | Reset at low battery | Input below dropout | Early shutdown | Low-dropout regulator; cut-off voltage |
| Sleep current too high | Battery drains with device off | Leakage; standby | Shelf-life loss | Load switch; low-leakage parts |
| Thermal test too short | Pass at 10 min, fail at 60 min | Time constant | Non-compliance | Test for 3 time constants |
| Battery ageing | Capacity falls after a year | Cycling; temperature | Life shortfall | Larger battery; thermal management |
| Charging hot | Cell temperature above 45 °C | High current; ambient | Ageing; safety | Temperature-controlled charging |

The most dangerous failure is thermal runaway. It is rare, but it can cause fire and injury. The protection circuit, the cell qualification, and the enclosure design must be treated as safety-critical, not as commodity components.

### 10.1 Ten red flags in power and thermal data

1. A sleep current higher than the datasheet's quiescent current.
2. A battery that discharges faster at low temperature than the model predicts.
3. A case temperature that rises after the display turns on.
4. A charger that terminates early or never terminates.
5. A cell voltage that rebounds after the device shuts down.
6. A thermal test shorter than three time constants.
7. A battery life claim without a temperature.
8. A device that resets during radio transmission.
9. A firmware update that runs while the battery is low.
10. A battery lot that is not recorded.

Each is a prompt to measure, not a conclusion.

### 10.2 What to do when the battery life falls short

1. Measure the current in every mode, including sleep.
2. Identify the largest unexpected load.
3. Check the regulator's quiescent current and the display's standby current.
4. Recompute the life with the measured loads.
5. Reduce the duty cycle or the load, or increase the battery.
6. Re-test the life at the worst-case temperature.

A shortfall is a measurement problem before it is a design problem. The measurement usually finds an omitted load.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The load table is complete | Battery life estimate | Omitted loads | Measure every mode |
| The duty cycle matches use | Average power | User behaviour | Field logs; worst-case profile |
| The battery capacity is usable | Life estimate | Derating; temperature; age | Discharge test at temperature |
| The regulator efficiency is known | Energy estimate | Load-dependent efficiency | Measure at the actual load |
| The thermal resistance is known | Temperature estimate | Enclosure; contact; airflow | Thermal test |
| The heat-transfer coefficient is representative | Temperature estimate | Still air versus forced | Test in the intended environment |
| The thermal capacitance is known | Time constant | Mass; materials | Step-response test |
| The ambient is representative | Temperature limit | Home; outdoor; transport | Environmental test |
| The applied-part contact is representative | Temperature limit | Contact area; pressure | Wear test |
| The battery protection is independent | Safety | Shared control loop | Fault injection |
| The charger profile matches the chemistry | Safety and life | Wrong voltage or current | Charge test; temperature |
| The device is used as intended | All estimates | Off-label use | Usability test; labelling |

The dominant assumption is that the load table is complete. It rarely is on the first pass. The missing loads are usually the display, the radio, the regulator quiescent current, and the sensor bias.

A second limit is that the thermal model is a single resistance. A real device has hot spots, internal air gaps, and contact resistances that the model ignores. The model tells you whether the design is close to the limit; the test tells you whether it passes.

A third limit is that the battery life claim is a model, not a measurement. A full discharge test at the worst-case temperature and profile is the evidence. A calculation is a prediction.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a datasheet?
- Which assumption was true on the bench but false in the home?
- Which assumption depends on the battery, the enclosure, or the ambient?
- Which assumption is encoded in a charger or protection circuit?
- Which assumption is specific to the validation profile?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Build a load table (recall, 20 minutes)

A device has an AFE at 0.4 mW continuous, an MCU at 4 mW for 10% of the time and 0.04 mW otherwise, a radio at 12 mW for 2% of the time and 0.002 mW otherwise, and a display at 15 mW for 20% of the time and 0.01 mW otherwise. Compute the average power.

**Deliverable:** a table and the total.

**Acceptance criteria:** each load's average computed; units; total.

**Time:** 20 minutes.

### Exercise 2. Compute battery life (application, 30 minutes)

A 200 mAh cell at 3.7 V powers the device in Exercise 1. The regulator efficiency is 88% and the derating is 20%. Compute the usable energy and the life in days.

**Deliverable:** a calculation and a one-sentence conclusion.

**Acceptance criteria:** nameplate and usable energy distinguished; life in days; units.

**Time:** 30 minutes.

### Exercise 3. Compute a thermal rise (application, 25 minutes)

A 1.5 W device has a 0.02 m² surface and a heat-transfer coefficient of 8 W/m²K. Compute the rise and the case temperature at a 35 °C ambient. Is it below the 43 °C limit?

**Deliverable:** a calculation and a compliance statement.

**Acceptance criteria:** correct rise; absolute temperature; comparison with the limit.

**Time:** 25 minutes.

### Exercise 4. Compute a time constant (application, 25 minutes)

The device in Exercise 3 has a thermal capacitance of 50 J/K. Compute the time constant and the time to 95% of the steady-state rise. How long should the thermal test run?

**Deliverable:** a calculation and a test duration.

**Acceptance criteria:** time constant; 3τ rule; test duration.

**Time:** 25 minutes.

### Exercise 5. Diagnose a battery shortfall (critique, 30 minutes)

A device predicted 6 days but delivers 4. The sleep current is 2.5 mA instead of 0.1 mA. Identify two likely causes and propose a fix for each.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** causes plausible; fixes specific; effect on life estimated.

**Time:** 30 minutes.

### Exercise 6. Design a thermal path (design, 35 minutes)

A 3 W processor in a sealed plastic handheld must keep the case below 43 °C at a 30 °C ambient. Propose a thermal path and estimate the required thermal resistance.

**Deliverable:** a design and a calculation.

**Acceptance criteria:** required resistance computed; path components named; safety considered.

**Time:** 35 minutes.

### Exercise 7. Write a verification plan (design, 40 minutes)

Write a power and thermal verification plan for the wearable in Section 6. Include the load measurement, the battery life test, the thermal soak, the applied-part temperature, and the charger test.

**Deliverable:** a plan with at least 8 rows.

**Acceptance criteria:** worst-case ambient and profile; test duration from the time constant; quantitative criteria.

**Time:** 40 minutes.

### Exercise 8. Evaluate a battery claim (critique, 25 minutes)

A vendor claims "7-day battery life." Write five questions that would test the claim.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover profile, temperature, derating, display and radio duty, and ageing; no question is answerable by the headline.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Load table | Table and total |
| 2 | Battery life | Calculation |
| 3 | Thermal rise | Calculation |
| 4 | Time constant | Test duration |
| 5 | Diagnosis | Causes and fixes |
| 6 | Thermal design | Path and resistance |
| 7 | Verification | Test plan |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** AFE: 0.4 mW. MCU: 4 × 0.10 + 0.04 × 0.90 = 0.436 mW. Radio: 12 × 0.02 + 0.002 × 0.98 = 0.242 mW. Display: 15 × 0.20 + 0.01 × 0.80 = 3.008 mW. Total = 4.086 mW. The display is 74% of the total.

**Answer 2 (Exercise 2).** Nameplate = 0.200 × 3.7 × 3600 = 2,664 J. Usable = 2,664 × 0.88 × 0.80 = 1,876 J. Daily = 4.086 mW × 86,400 = 353.0 J. Life = 1,876 / 353.0 = 5.31 days.

**Answer 3 (Exercise 3).** Rise = 1.5 / (8 × 0.02) = 9.375 K. Case = 35 + 9.375 = 44.4 °C, above the 43 °C limit. The device is not compliant at a 35 °C ambient without a thermal design change.

**Answer 4 (Exercise 4).** R_th = 1/(8 × 0.02) = 6.25 K/W. τ = 6.25 × 50 = 312.5 s = 5.21 min. 95% of the rise takes 3τ = 15.6 min. The test should run for at least 20 minutes to be safe.

**Answer 5 (Exercise 5).** Causes: a regulator with a high quiescent current (for example, 2 mA), or a display or radio that remains in a standby mode instead of shutting down. Fixes: replace the regulator with a low-quiescent part (for example, 20 µA) and add a load switch to the display. The 2.4 mA difference at 3.7 V is 8.9 mW, which is more than twice the average power of the design and explains the shortfall.

**Answer 6 (Exercise 6).** Required R_th = (43 − 30)/3 = 4.33 K/W. A metal case with a thermal pad from the processor to the case, plus a heat spreader, can achieve 3–5 K/W. The case must be electrically isolated from the patient, and the contact area must be considered. If the resistance cannot be achieved, the power must be reduced.

**Answer 7 (Exercise 7).** The plan: measure the current in each mode with the production firmware; run a full discharge at 15, 25, and 40 °C; run a 60-minute thermal soak in the production enclosure at 40 °C ambient; measure the applied-part temperature at the contact surface; test the charger at 0, 25, and 45 °C; test a brown-out at the cut-off voltage. Acceptance: average power < 3.0 mW; life > 5 days; case < 43 °C; no overcharge; safe reset.

**Answer 8 (Exercise 8).** Questions: (1) What display and radio duty cycles does the claim assume? (2) At what ambient temperature? (3) What derating for battery age and discharge rate? (4) Does the claim include the sleep current of the display and regulator? (5) Is the life measured or calculated? A headline cannot answer any of them.

**Rubric for the design exercises.** A strong design quantifies the load table, the derating, the thermal resistance, the time constant, and the verification. A weak design quotes a battery capacity and assumes the rest.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: average power | 4.09 mW | ±0.05 mW |
| Exercise 2: usable energy | 1,876 J | ±20 J |
| Exercise 2: life | 5.31 days | ±0.1 day |
| Exercise 3: rise | 9.38 K | ±0.2 K |
| Exercise 3: case | 44.4 °C | ±0.3 °C |
| Exercise 4: time constant | 5.21 min | ±0.1 min |
| Exercise 4: 3τ | 15.6 min | ±0.3 min |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 10 | 5 | 20 |
| 2 | 5 | 15 | 10 | 30 |
| 3 | 5 | 10 | 10 | 25 |
| 4 | 5 | 10 | 10 | 25 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 20 | 10 | 35 |
| 7 | 5 | 25 | 10 | 40 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. How do you estimate battery life?**
A strong answer builds a duty-cycle table, applies the derating and regulator efficiency, and states the conditions. A weak answer divides capacity by current.

**2. Why is the usable battery capacity lower than the nameplate?**
A strong answer names discharge rate, temperature, cut-off voltage, and ageing. A weak answer says "efficiency".

**3. How do you compute a thermal rise?**
A strong answer uses power divided by the heat-transfer coefficient times the area, and states the assumptions. A weak answer says "it gets hot".

**4. What is the thermal time constant, and why does it matter?**
A strong answer defines it as R_th × C_th and uses it to set the test duration. A weak answer confuses it with the rise time.

**5. What are the IEC 60601-1 applied-part temperature limits?**
A strong answer states the 41 °C disclosure threshold and the 43 °C limit for 10 minutes or more, and notes the vulnerable groups. A weak answer quotes one number.

**6. What makes a lithium battery safe in a medical device?**
A strong answer names the cell qualification, the independent protection circuit, the charger profile, and the temperature monitoring. A weak answer says "use a protected cell".

**7. Why does the display dominate many wearable power budgets?**
A strong answer computes the display's average power from its duty cycle and compares it with the AFE and radio. A weak answer assumes the radio dominates.

**8. How do you test a thermal design?**
A strong answer uses the production enclosure, the worst-case ambient, the worst-case workload, and a duration of at least three time constants. A weak answer tests on the bench for 10 minutes.

**9. What happens to a device at a low battery voltage?**
A strong answer explains the regulator dropout, the brown-out, and the safe shutdown. A weak answer says the device "turns off".

**10. What would make you reject a power design?**
A strong answer names a battery life shortfall at the worst case, a thermal limit exceedance, a missing protection circuit, or an unmeasured quiescent load. A weak answer rejects on battery size alone.

**11. How does temperature affect battery life?**
A strong answer explains the capacity loss at low temperature, the ageing at high temperature, and the internal resistance rise. A weak answer says "cold drains the battery".

**12. Why is the thermal time constant important for testing?**
A strong answer uses it to set the test duration and to distinguish transient from steady state. A weak answer measures for a fixed 10 minutes.

**13. How do you choose between a linear and a switching regulator?**
A strong answer compares efficiency at the actual load, quiescent current, noise, and dropout. A weak answer chooses switching because it is more efficient.

**14. What is the difference between the 41 °C and 43 °C limits?**
A strong answer explains that 41 °C triggers disclosure and 43 °C is the limit for prolonged contact, and that vulnerable groups may be harmed below 43 °C. A weak answer treats them as the same.

## 15. Key terms

- **Ambient temperature** — temperature of the surrounding air.
- **Applied part** — part of the device in contact with the patient.
- **Battery capacity** — charge stored, in mAh.
- **Brown-out** — supply drop that can reset the device.
- **Case temperature** — temperature of the device enclosure.
- **Coulombic efficiency** — charge out divided by charge in.
- **Cut-off voltage** — voltage at which the device stops.
- **Derating** — reduction from nameplate capacity.
- **Duty cycle** — fraction of time a load is active.
- **Energy** — capacity to do work, in joules.
- **Energy harvesting** — conversion of ambient energy.
- **Heat-transfer coefficient** — heat flux per area per temperature difference.
- **Internal resistance** — resistance of the cell.
- **Junction temperature** — temperature of the silicon.
- **Leakage current** — current through insulation.
- **Nameplate capacity** — rated capacity at a specified rate.
- **Overcurrent protection** — circuit that limits current.
- **Overvoltage protection** — circuit that limits voltage.
- **Peukert effect** — capacity falls at higher discharge rates.
- **Power** — rate of energy use, in watts.
- **Quiescent current** — current drawn when idle.
- **Regulator efficiency** — output power divided by input power.
- **Self-discharge** — loss of charge over time.
- **State of charge** — fraction of capacity remaining.
- **State of health** — capacity relative to new.
- **Thermal capacitance** — energy to raise temperature.
- **Thermal interface material** — material that fills a gap between hot surfaces.
- **Thermal resistance** — temperature rise per watt.
- **Thermal runaway** — self-accelerating heating.
- **Thermal time constant** — product of thermal resistance and capacitance.
- **Thermoelectric generator** — device that converts heat to electricity.
- **Arrhenius model** — temperature-accelerated ageing model.
- **Bulk capacitor** — capacitor that supplies pulse current.
- **Charge cycle** — one full charge and discharge.
- **Depth of discharge** — fraction of capacity used.
- **Load switch** — switch that disconnects a load.
- **Power path** — circuit that manages battery and external power.
- **Thermal break** — low-conductivity path between hot and contact surfaces.
- **Vent** — pressure-relief feature in a cell or enclosure.
- **Charge rate** — current divided by capacity, in C.
- **End-of-life** — capacity at which the battery no longer meets the requirement.
- **Power density** — power per unit mass or volume.
- **Specific heat** — energy per unit mass per kelvin.
- **Thermal mass** — another term for thermal capacitance.
- **Vent path** — enclosure feature that directs a cell's vented gas.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards and regulation**

1. IEC 60601-1:2005+AMD1:2012+AMD2:2020, *General requirements for basic safety and essential performance*, clause 11.1.2.2 and Tables 23–24. https://webstore.iec.ch/en/publication/26169
2. IEC 60601-1-11:2015+AMD1:2020, *Requirements for medical electrical equipment used in the home healthcare environment*. https://webstore.iec.ch/en/publication/67384
3. IEC 62133-2:2017+AMD1:2021, *Secondary cells and batteries — Safety requirements for portable sealed secondary cells — Part 2: Lithium systems*. https://webstore.iec.ch/en/publication/32662
4. UN 38.3, *Transport of dangerous goods — Lithium batteries*. https://unece.org/transport/dangerous-goods
5. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
6. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
7. IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances*. https://webstore.iec.ch/en/publication/67554
8. U.S. Food and Drug Administration, recognized consensus standards database, IEC 62133-2 recognition 19-55, 2025. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/search.cfm

**Background**

9. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
10. D. Linden and T. B. Reddy, *Handbook of Batteries*, 3rd ed., McGraw-Hill, 2001.
11. J. Larminie and A. Dicks, *Fuel Cell Systems Explained*, 2nd ed., Wiley, 2003.
12. T. L. Bergman et al., *Fundamentals of Heat and Mass Transfer*, 8th ed., Wiley, 2018.
13. IEC 60068-2-1, *Environmental testing — Cold*. https://webstore.iec.ch/
14. IEC 60068-2-2, *Environmental testing — Dry heat*. https://webstore.iec.ch/

**Device evidence**

15. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
16. IEC 60601-1:2005, Annex A rationale on applied-part temperatures. https://webstore.iec.ch/en/publication/26169
17. IEC 62133-2:2017, test clauses 7.1–7.3. https://webstore.iec.ch/en/publication/32662
18. U.S. Department of Energy, *Battery Storage Handbook* (for chemistry background). https://www.energy.gov/eere/articles/battery-storage-handbook
19. IEC 60601-1:2005, clause 11.1.2.2, applied-part temperature disclosure.
20. IEC 62133-2:2017, test clauses 7.1–7.3, battery safety tests.
21. J. G. Webster, *Medical Instrumentation*, chapter on electrical safety.
22. T. L. Bergman et al., *Fundamentals of Heat and Mass Transfer*, 8th ed.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_socs-npus-and-gpus.md`](06_socs-npus-and-gpus.md)
- Next topic: [`08_firmware-and-rtos.md`](08_firmware-and-rtos.md)
- Related: [`05_mcus-dsps-and-fpgas.md`](05_mcus-dsps-and-fpgas.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Lab on device teardown: see Lab 14A.6 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
- **Capacity fade** — loss of capacity with cycling and age.
- **Cell balancing** — equalising cells in a series pack.
- **C-rate** — charge or discharge rate relative to capacity.
- **Energy density** — energy per unit mass or volume.
- **Float voltage** — voltage held after charge.
- **Gas gauge** — circuit that estimates state of charge.
- **Internal short** — fault inside a cell.
- **Over-discharge** — discharge below the safe voltage.
- **Pack** — multiple cells with protection and management.
- **Primary cell** — non-rechargeable cell.
- **Secondary cell** — rechargeable cell.
- **State of charge (SoC)** — remaining charge as a fraction.
- **Thermal management** — design that keeps temperatures in range.

Further sources:

23. IEC 60068-2-1, cold environmental test.
24. IEC 60068-2-2, dry heat environmental test.
25. UN 38.3, lithium battery transport tests.
26. IEC 62133-2:2017, battery safety tests.
27. IEC 60601-1:2005, clause 11.1.2.2.

- **Thermal runaway propagation** — spread of a cell failure to neighbouring cells.
- **Vent gas** — gas released by a failing cell.
- **Charge termination** — end of the charging profile.
- **Battery management system** — circuit that monitors and protects the pack.
- **Thermal fuse** — one-time disconnect at a temperature limit.
