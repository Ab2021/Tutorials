# Day 99: Wireless Communication Overview
## Phase 1: Core Embedded Engineering Foundations | Week 15: Wireless Communication Basics

---

> **📝 Content Creator Instructions:**
> This document is designed to produce **comprehensive, industry-grade educational content**. 
> - **Target Length:** The final filled document should be approximately **1000+ lines** of detailed markdown.
> - **Depth:** Do not skim over details. Explain *why*, not just *how*.
> - **Structure:** If a topic is complex, **DIVIDE IT INTO MULTIPLE PARTS** (Part 1, Part 2, etc.).
> - **Code:** Provide complete, compilable code examples, not just snippets.
> - **Visuals:** Use Mermaid diagrams for flows, architectures, and state machines.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the fundamentals of Radio Frequency (RF) communication (Frequency, Wavelength, Power).
2.  **Differentiate** between common modulation techniques (ASK, FSK, PSK, LoRa).
3.  **Compare** wireless standards (BLE, WiFi, Zigbee, LoRaWAN) based on Range, Data Rate, and Power.
4.  **Calculate** Link Budget basics (Tx Power, Path Loss, Rx Sensitivity).
5.  **Identify** RF components on a PCB (Antenna, Matching Network, Crystal).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Various RF Modules (optional visualization): HC-05, ESP8266, nRF24L01.
*   **Software Required:**
    *   None (Theory day).
*   **Prior Knowledge:**
    *   Day 22 (UART) - Most modules use UART.
    *   Day 29 (SPI) - Some use SPI.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: RF Physics 101
*   **Frequency ($f$):** Cycles per second (Hz). 2.4 GHz = 2.4 Billion cycles/sec.
*   **Wavelength ($\lambda$):** Distance a wave travels in one cycle. $\lambda = c / f$.
    *   @ 2.4 GHz, $\lambda \approx 12.5$ cm.
    *   **Antenna Size:** Usually $\lambda / 4$ (Quarter Wave Monopole) $\approx 3.1$ cm.
*   **Power (dBm):** Logarithmic scale relative to 1mW.
    *   0 dBm = 1 mW.
    *   10 dBm = 10 mW.
    *   20 dBm = 100 mW (Max for WiFi usually).
    *   -100 dBm = 0.1 pW (Very weak signal).

### 🔹 Part 2: Modulation
How to send 1s and 0s over a sine wave?
1.  **ASK (Amplitude Shift Keying):** High Amplitude = 1, Low = 0. (Simple, Noise sensitive).
2.  **FSK (Frequency Shift Keying):** High Freq = 1, Low Freq = 0. (Bluetooth uses GFSK).
3.  **PSK (Phase Shift Keying):** Phase shift = 1. (Zigbee uses O-QPSK).
4.  **LoRa (Chirp Spread Spectrum):** Rising chirp = 1. (Long Range, Noise immune).

### 🔹 Part 3: The Wireless Landscape
| Standard | Freq | Range | Data Rate | Power | Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WiFi** | 2.4/5GHz | 50m | > 100 Mbps | High | Video, Internet |
| **BLE** | 2.4GHz | 100m | 1 Mbps | Low | Wearables |
| **Zigbee** | 2.4GHz | 100m | 250 kbps | Low | Mesh, Smart Home |
| **LoRa** | 868/915MHz | > 10km | < 50 kbps | Very Low | Agriculture, Metering |
| **NFC** | 13.56MHz | 4cm | 424 kbps | Low | Payment, Access |

---

## 💻 Implementation: Link Budget Calculator

> **Instruction:** Create a C function to estimate range.

### 👨‍💻 Code Implementation

#### Step 1: Friis Transmission Equation
$P_{rx} = P_{tx} + G_{tx} + G_{rx} - L_{path}$
*   $L_{path}$ (Free Space Path Loss) $= 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10}(4\pi/c)$.
*   Simplified for 2.4GHz (d in meters): $L_{path} \approx 40 + 20 \log_{10}(d)$.

#### Step 2: C Function
```c
#include <math.h>

// Calculate Max Range in Meters
// tx_power_dbm: e.g., 0 (1mW)
// sensitivity_dbm: e.g., -90
// freq_mhz: e.g., 2400
float Calculate_Range(float tx_power_dbm, float sensitivity_dbm, float freq_mhz) {
    // Max Path Loss allowed
    float max_path_loss = tx_power_dbm - sensitivity_dbm;
    
    // FSPL = 20log(d) + 20log(f) - 27.55 (f in MHz, d in m)
    // max_path_loss = 20log(d) + 20log(freq) - 27.55
    // 20log(d) = max_path_loss - 20log(freq) + 27.55
    // log(d) = (max_path_loss - 20log(freq) + 27.55) / 20
    
    float term1 = max_path_loss;
    float term2 = 20.0f * log10f(freq_mhz);
    float term3 = 27.55f;
    
    float log_d = (term1 - term2 + term3) / 20.0f;
    
    return powf(10.0f, log_d);
}
```

#### Step 3: Test
```c
void Test_Range(void) {
    // BLE: 0dBm Tx, -90dBm Rx, 2400MHz
    float range_ble = Calculate_Range(0, -90, 2400);
    printf("BLE Range: %.2f m\n", range_ble); // ~100m
    
    // LoRa: 14dBm Tx, -130dBm Rx, 915MHz
    float range_lora = Calculate_Range(14, -130, 915);
    printf("LoRa Range: %.2f m\n", range_lora); // ~10km
}
```

---

## 🔬 Lab Exercise: Lab 99.1 - RSSI Scanner

### 1. Lab Objectives
- Use an ESP8266/ESP32 (if available) or just simulate.
- Scan for WiFi networks.
- Print SSID and RSSI (Received Signal Strength Indicator).

### 2. Step-by-Step Guide

#### Phase A: Simulation (STM32)
Since STM32F4 doesn't have radio, we will simulate the output of a scanner.
```c
typedef struct {
    char ssid[32];
    int rssi;
} WifiAP_t;

WifiAP_t ap_list[] = {
    {"Home_WiFi", -50},
    {"Neighbor_WiFi", -85},
    {"Coffee_Shop", -70}
};

void Scan_WiFi(void) {
    printf("Scanning...\n");
    for(int i=0; i<3; i++) {
        printf("SSID: %s, RSSI: %d dBm ", ap_list[i].ssid, ap_list[i].rssi);
        
        if (ap_list[i].rssi > -60) printf("[Excellent]\n");
        else if (ap_list[i].rssi > -70) printf("[Good]\n");
        else if (ap_list[i].rssi > -80) printf("[Fair]\n");
        else printf("[Poor]\n");
    }
}
```

#### Phase B: Run
1.  Run code.
2.  **Observation:** Understand that -50 is stronger than -85.

### 3. Verification
RSSI is always negative. Closer to 0 is better. -100 is usually the noise floor.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Antenna Tuning (Theory)
- **Goal:** Understand VSWR.
- **Task:**
    1.  If Antenna impedance (50 Ohm) doesn't match Chip impedance, power reflects back.
    2.  **VSWR (Voltage Standing Wave Ratio):** 1:1 is perfect. 2:1 is bad.
    3.  Touch the antenna of a radio. Signal drops. Why? You detuned it (added capacitance).

### Lab 3: Packet Error Rate (PER)
- **Goal:** Quality Metric.
- **Task:**
    1.  Send 1000 packets.
    2.  Count how many ACKed.
    3.  PER = (1000 - ACK) / 1000.
    4.  If PER > 1%, link is unstable.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Short Range
*   **Cause:** Antenna inside metal enclosure (Faraday Cage).
*   **Solution:** Use external antenna or plastic case.

#### 2. Interference
*   **Cause:** Microwave Ovens, Bluetooth, and WiFi all share 2.4GHz.
*   **Solution:** Use 5GHz or Sub-GHz (LoRa) for critical data.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Retry Logic:** Wireless is unreliable. Always implement Retries (ARQ - Automatic Repeat Request). Don't just send and forget.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does 900MHz go further than 2.4GHz?
    *   **A:** Lower frequency has longer wavelength, which penetrates obstacles better and has lower path loss.
2.  **Q:** What is "Sensitivity"?
    *   **A:** The minimum signal power the receiver needs to decode data. -100dBm is better than -90dBm.

### Challenge Task
> **Task:** "Spectrum Analyzer". If you have an SDR (Software Defined Radio) dongle ($20), plug it into PC. Use "SDR#" software to visualize the 433MHz or 900MHz ISM bands. See the invisible signals!

---

## 📚 Further Reading & References
- [TI: ISM Band Basics](https://www.ti.com/lit/an/swra048/swra048.pdf)

---
