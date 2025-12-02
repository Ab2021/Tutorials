# Day 173: Compliance & Certification (CE, FCC, RoHS)
## Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance

---

## 🎯 Learning Objectives
1.  **Navigate** the regulatory landscape: CE (Europe), FCC (USA), UKCA (UK), CCC (China).
2.  **Understand** Environmental Compliance: RoHS (Hazardous Substances), REACH (Chemicals), WEEE (Recycling).
3.  **Prepare** a Technical File for certification.
4.  **Execute** Pre-Compliance Testing to avoid costly failures at the test house.
5.  **Label** your product correctly (Logos, IDs).

---

## 📚 Prerequisites & Preparation
*   **Context:** You cannot legally sell your camera without these marks.
*   **Documents:** Declaration of Conformity (DoC) templates.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Electromagnetic Compatibility (EMC)
*   **FCC (Federal Communications Commission):**
    *   **Part 15 Class A:** Industrial (Relaxed limits).
    *   **Part 15 Class B:** Residential (Strict limits).
    *   Focuses mainly on **Emissions** (Don't jam the radio).
*   **CE (Conformité Européenne):**
    *   **RED (Radio Equipment Directive):** If you have WiFi/Bluetooth.
    *   **EMC Directive:** If no radio.
    *   Requires both **Emissions** AND **Immunity** (Must survive ESD, Surge).

### 🔹 Part 2: Environmental (Green)
*   **RoHS (Restriction of Hazardous Substances):** No Lead (Pb), Mercury, Cadmium.
    *   **Impact:** You must use Lead-Free Solder (SAC305).
*   **REACH:** Registration of Chemicals. (e.g., Plasticizers in cables).
*   **WEEE:** Waste Electrical and Electronic Equipment. You pay a fee to fund recycling.

### 🔹 Part 3: Safety
*   **LVD (Low Voltage Directive):** For > 50V AC.
*   **IEC 62368-1:** Audio/Video/IT equipment safety. (Fire, Shock, Thermal burns).

---

## 💻 Implementation Examples

### Example 1: Declaration of Conformity (DoC)

The document you sign.

```text
EU DECLARATION OF CONFORMITY

Manufacturer: MyCamera Inc.
Address: 123 Tech Park.

Product: Smart Security Camera
Model: SC-100

We declare under our sole responsibility that the product is in conformity with:
1. EMC Directive 2014/30/EU
   - Standards: EN 55032 (Emission), EN 55035 (Immunity)
2. RoHS Directive 2011/65/EU

Signed:
John Doe, CTO
Date: 2023-10-27
```

### Example 2: Software Config for FCC

If you have WiFi, you must limit power based on region.

```bash
# In /etc/modprobe.d/wifi.conf
# Set Regulatory Domain to US (FCC)
options cfg80211 cfg80211_regulatory_domain=US

# Verify
iw reg get
# country US: DFS-FCC
# (2402 - 2472 @ 40.000 KHz), (N/A, 3000 mBm), (N/A)
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Component Audit (RoHS)

**Objective:** Ensure supply chain compliance.

**Steps:**
1.  Export BOM (Bill of Materials).
2.  Check every datasheet. Look for the "RoHS Compliant" logo.
3.  **Risk:** Old components or cheap connectors from unknown sources might contain Lead.
4.  **Action:** Request "Material Declaration" (IPC-1752) from suppliers.

### Lab 2: Label Design

**Objective:** Fit everything on a small sticker.

**Steps:**
1.  **Mandatory:** FCC ID, CE Logo, WEEE (Trash can with X), Model Number, Power Rating (5V 1A).
2.  **Size:** If the device is too small, some text can go in the manual, but the FCC ID usually must be on the device.
3.  **Durability:** The label must survive the "Rub Test" (Water and Petroleum).

### Lab 3: Pre-Compliance ESD Test

**Objective:** Zap it.

**Steps:**
1.  Use a Piezo lighter (from a BBQ lighter) if you don't have an ESD gun (poor man's test).
2.  Zap the USB port shell (Ground).
3.  **Observe:** Does the video freeze? Does it reboot?
4.  **Fix:** Add TVS diodes on USB D+/D-. Use a better shielded case.

---

## 🐛 Debugging Certification Failures

### Debug 1: "Radiated Emission Failure at 2.4GHz"

**Symptom:** Failing FCC Part 15.

**Cause:**
*   WiFi antenna harmonics or CPU clock harmonics.
*   **Fix:** Add a shield can. Add copper tape. Check antenna matching.

### Debug 2: "Surge Test Failure"

**Symptom:** Device dies during CE Surge Test (1kV).

**Cause:**
*   Input protection insufficient.
*   **Fix:** Add a MOV (Metal Oxide Varistor) or GDT (Gas Discharge Tube) at the power input.

---

## ⚡ Performance Optimization

### Optimization 1: Modular Certification

*   Use a pre-certified WiFi Module (e.g., ESP32, Murata).
*   **Benefit:** You inherit their FCC ID. You only need to test "Unintentional Radiator" (much cheaper).
*   **Cost:** Modules are more expensive than chip-down design.

### Optimization 2: Series Certification

*   Certify the "Worst Case" model (Max CPU, Max WiFi Power).
*   Derive other models (Lower CPU, No WiFi) as "Family Variants".
*   Saves paying for full testing on every SKU.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the "CE" mark?** (Self-declaration that the product meets EU safety/EMC rules. Not a quality mark).
2.  **What is "FCC ID"?** (A unique code assigned by the FCC. Allows tracking the device owner/manufacturer).
3.  **Why is Lead (Pb) banned?** (Neurotoxin. Leaches into groundwater from landfills).

### Practical Challenges

1.  **User Manual:** Write the "FCC Statement" section. "This device complies with Part 15... Operation is subject to...".
2.  **Customs:** What happens if you ship a non-CE marked camera to Germany? (Customs will seize and destroy it).

---

## 📚 Further Reading & Resources

### Documentation
*   **FCC OET Knowledge Database (KDB).**
*   **"The Blue Guide" (EU Product Rules).**

---

## 🎓 Summary

Today we covered:
- ✅ **CE/FCC:** The passport for your product.
- ✅ **RoHS/WEEE:** Green electronics.
- ✅ **DoC:** The legal promise.
- ✅ **Labeling:** What must be printed.
- ✅ **Pre-Compliance:** Testing early.

**Next:** Day 174 - Week 27 Review & Project (Validation Suite).

---

**Day 173 Complete** | Phase 3: Camera Systems & ISP | Week 27: Testing, Validation & Compliance


