# Day 225: Wireless Subsystem (mac80211 & cfg80211)
## Phase 2: Linux Kernel & Device Drivers | Week 34: Wireless & Embedded Networking

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
1.  **Understand** the Linux Wireless Architecture (`cfg80211` vs `mac80211`).
2.  **Explain** the difference between SoftMAC and FullMAC drivers.
3.  **Navigate** the `wiphy` (Wireless Physical Device) structure.
4.  **Register** a dummy wireless device.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
*   **Software Required:**
    *   Kernel Source.
    *   `iw`, `hostapd`.
*   **Prior Knowledge:**
    *   Day 170 (Net Dev Basics).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Architecture
*   **Userspace:** `hostapd` (AP mode), `wpa_supplicant` (Station mode), `iw` (Configuration).
*   **Netlink (nl80211):** Communication protocol between userspace and kernel.
*   **cfg80211:** The configuration layer. Handles regulatory domains, channel selection, and management frames.
*   **mac80211:** The SoftMAC layer. Implements the 802.11 state machine (scanning, auth, assoc) in software for devices that only handle raw frame transmission.
*   **Driver:** Talks to hardware.

### 🔹 Part 2: SoftMAC vs FullMAC
*   **SoftMAC:** Hardware sends/receives raw frames. Kernel does the heavy lifting (802.11 management). Most USB/PCI dongles (Atheros, Realtek) are SoftMAC.
*   **FullMAC:** Hardware/Firmware handles 802.11 management. Kernel just sends Ethernet frames. Intel WiFi, Embedded IoT chips are often FullMAC.

---

## 💻 Implementation: Registering a Wiphy

> **Instruction:** Create a dummy module that registers a wireless device using `cfg80211`.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <net/cfg80211.h>

static struct wiphy *my_wiphy;

static const struct cfg80211_ops my_ops = {
    // Minimal ops required
    .start_radar_detection = NULL,
};

static int __init my_wifi_init(void) {
    int ret;
    
    // 1. Allocate Wiphy
    my_wiphy = wiphy_new(&my_ops, sizeof(void *));
    if (!my_wiphy) return -ENOMEM;
    
    // 2. Configure Capabilities
    strcpy(my_wiphy->fw_version, "1.0");
    strcpy(my_wiphy->driver_version, "1.0");
    my_wiphy->max_scan_ssids = 4;
    my_wiphy->max_scan_ie_len = 1000;
    
    // 3. Register
    ret = wiphy_register(my_wiphy);
    if (ret) {
        wiphy_free(my_wiphy);
        return ret;
    }
    
    pr_info("MyWifi: Registered wiphy%d\n", my_wiphy->idx);
    return 0;
}

static void __exit my_wifi_exit(void) {
    wiphy_unregister(my_wiphy);
    wiphy_free(my_wiphy);
}

module_init(my_wifi_init);
module_exit(my_wifi_exit);
MODULE_LICENSE("GPL");
```

---

## 🔬 Lab Exercise: Lab 225.1 - Inspecting with `iw`

### 1. Lab Objectives
- Verify registration.
- Check capabilities.

### 2. Step-by-Step Guide
1.  **Load Module:** `insmod mywifi.ko`.
2.  **List Devices:**
    ```bash
    iw list
    ```
    *   You should see `Wiphy <N>` with the details we set.
3.  **Check Dmesg:**
    *   "MyWifi: Registered wiphy..."

---

## 🧪 Additional / Advanced Labs

### Lab 2: Adding Bands (2.4GHz / 5GHz)
- **Goal:** Advertise supported channels.
- **Task:**
    *   Populate `my_wiphy->bands[NL80211_BAND_2GHZ]`.
    *   Define channels (freq, power limits).
    *   `iw list` will now show supported frequencies.

### Lab 3: Regulatory Hints
- **Goal:** Set country code.
- **Task:**
    *   `wiphy_apply_custom_regulatory(my_wiphy, &reg_request)`.
    *   Or use `crda` (Central Regulatory Domain Agent) from userspace.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Registration Fail
*   **Cause:** Missing mandatory ops or invalid band configuration.
*   **Fix:** Check `dmesg` for specific complaints from `cfg80211`.

---

## ⚡ Optimization & Best Practices

### Hardware Encryption
*   If HW supports AES/CCMP, advertise `NL80211_FEATURE_DATA_ACK` and implement `set_key` ops.
*   Otherwise, kernel does SW encryption (slow).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `nl80211`?
    *   **A:** The Netlink family used to configure wireless devices. It replaced the old `Wireless Extensions` (WE/WEXT).
2.  **Q:** Does `mac80211` handle Roaming?
    *   **A:** Yes, for SoftMAC drivers, `mac80211` (along with `wpa_supplicant`) handles the roaming logic (scanning, re-associating).

### Challenge Task
> **Task:** "The Fake AP".
> *   Configure your dummy driver to support AP mode (`NL80211_IFTYPE_AP`).
> *   Try to start `hostapd` on it. (It will fail later when trying to send beacons, but initialization should pass).

---

## 📚 Further Reading & References
- [Linux Wireless Wiki](https://wireless.wiki.kernel.org/)
- [mac80211 Documentation](https://www.kernel.org/doc/html/latest/driver-api/80211/index.html)

---
