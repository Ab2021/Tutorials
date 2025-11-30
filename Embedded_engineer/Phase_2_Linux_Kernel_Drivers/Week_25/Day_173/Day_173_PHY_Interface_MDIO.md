# Day 173: PHY Interface (MDIO/MII)
## Phase 2: Linux Kernel & Device Drivers | Week 25: Network Device Drivers

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
1.  **Explain** the MAC vs PHY architecture and the MDIO bus.
2.  **Register** an MDIO bus driver (`mdiobus_register`).
3.  **Connect** to a PHY using the PHY Lib (`phy_connect`, `phy_start`).
4.  **Handle** Link State changes (Speed/Duplex updates) in the callback.
5.  **Read/Write** PHY registers using `mdio-tool` or `mii-tool`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `ethtool`.
*   **Prior Knowledge:**
    *   Day 170 (Netdev).
    *   Day 142 (I2C - MDIO is similar).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: MAC vs PHY
*   **MAC (Media Access Control):** The digital logic inside the SoC/NIC. Handles Ethernet framing, DMA.
*   **PHY (Physical Layer):** The analog chip. Handles the cable (Copper/Fiber), encoding, link negotiation.
*   **Interface:**
    *   **Data:** MII, RMII, RGMII (Parallel data).
    *   **Control:** MDIO (Management Data Input/Output). 2 wires: MDC (Clock), MDIO (Data).

### 🔹 Part 2: Linux PHY Lib
Linux abstracts the PHY. The MAC driver doesn't need to know if it's a Realtek or Atheros PHY.
1.  **MDIO Bus:** Scans for PHYs (Addresses 0-31).
2.  **PHY Driver:** Generic or specific (e.g., `realtek.c`).
3.  **State Machine:** Handles Auto-negotiation, Link Up/Down.

---

## 💻 Implementation: MDIO Bus Registration

> **Instruction:** If your MAC has an internal MDIO controller, you must register it.

### 👨‍💻 Code Implementation

#### Step 1: Structure
```c
#include <linux/phy.h>

struct snull_priv {
    // ...
    struct mii_bus *mii_bus;
    struct phy_device *phydev;
};
```

#### Step 2: MDIO Read/Write Ops
```c
static int snull_mdio_read(struct mii_bus *bus, int phy_id, int regnum) {
    struct snull_priv *priv = bus->priv;
    // Write PHY_ID and REG to HW register
    // Wait for Done
    // Return Data
    return 0xFFFF; // Dummy
}

static int snull_mdio_write(struct mii_bus *bus, int phy_id, int regnum, u16 val) {
    // Write to HW
    return 0;
}
```

#### Step 3: Registration (in Probe)
```c
priv->mii_bus = mdiobus_alloc();
priv->mii_bus->name = "snull_mdio";
snprintf(priv->mii_bus->id, MII_BUS_ID_SIZE, "%s-%x", "snull", 0);
priv->mii_bus->read = snull_mdio_read;
priv->mii_bus->write = snull_mdio_write;
priv->mii_bus->priv = priv;

// Scan for PHYs
ret = mdiobus_register(priv->mii_bus);
```

---

## 💻 Implementation: Connecting to the PHY

> **Instruction:** Once the bus is up, connect the netdev to a specific PHY.

### 👨‍💻 Code Implementation

#### Step 1: Link Change Callback
Called by PHY Lib when link status changes.
```c
static void snull_adjust_link(struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    struct phy_device *phydev = priv->phydev;
    
    if (phydev->link) {
        if (priv->link_up != 1) {
            priv->link_up = 1;
            netif_carrier_on(dev);
            pr_info("%s: Link Up - %d/%s\n", dev->name, 
                    phydev->speed, 
                    phydev->duplex == DUPLEX_FULL ? "Full" : "Half");
            
            // Configure MAC hardware for new speed
            // hw_set_speed(priv, phydev->speed);
        }
    } else {
        if (priv->link_up != 0) {
            priv->link_up = 0;
            netif_carrier_off(dev);
            pr_info("%s: Link Down\n", dev->name);
        }
    }
}
```

#### Step 2: Connect (in Open)
```c
static int snull_open(struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    
    // 1. Find PHY (e.g., "stmmac-0:01")
    // Or use phy_find_first(priv->mii_bus);
    
    // 2. Connect
    // Interface: PHY_INTERFACE_MODE_RGMII
    ret = phy_connect_direct(dev, priv->phydev, snull_adjust_link, 
                             PHY_INTERFACE_MODE_RGMII);
    
    // 3. Start State Machine
    phy_start(priv->phydev);
    
    return 0;
}
```

#### Step 3: Disconnect (in Stop)
```c
static int snull_stop(struct net_device *dev) {
    struct snull_priv *priv = netdev_priv(dev);
    
    phy_stop(priv->phydev);
    phy_disconnect(priv->phydev);
    
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 173.1 - Fixed PHY

### 1. Lab Objectives
- Since we don't have a real PHY in QEMU (usually), use the "Fixed PHY" emulation.
- Register a fixed link.

### 2. Step-by-Step Guide
1.  **Device Tree:**
    ```dts
    ethernet@... {
        phy-mode = "rgmii";
        fixed-link {
            speed = <1000>;
            full-duplex;
        };
    };
    ```
2.  **Driver:** Use `of_phy_get_and_connect(dev, np, snull_adjust_link)`.
3.  **Verify:** `dmesg` should show "Link Up - 1000/Full".

---

## 🧪 Additional / Advanced Labs

### Lab 2: Ethtool Integration
- **Goal:** Allow user to check link status via `ethtool`.
- **Task:**
    1.  Implement `get_link_ksettings` and `set_link_ksettings`.
    2.  Delegate to `phy_ethtool_get_link_ksettings(dev->phydev, cmd)`.
    3.  Run `ethtool sn0`.

### Lab 3: MDIO Bitbanging
- **Goal:** Implement MDIO over GPIOs.
- **Task:**
    1.  Use `alloc_mdio_bitbang`.
    2.  Provide `set_mdc`, `set_mdio_dir`, `set_mdio_data`, `get_mdio_data`.
    3.  Linux handles the timing.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No PHY found"
*   **Cause:** Wrong MDIO address (0-31).
*   **Cause:** PHY in reset (Check Reset GPIO).
*   **Cause:** MDIO Clock too fast (Max 2.5 MHz).

#### 2. Link Flapping (Up/Down/Up)
*   **Cause:** Auto-negotiation failing.
*   **Cause:** Cable issues.
*   **Fix:** Force speed via ethtool to debug.

---

## ⚡ Optimization & Best Practices

### `phy_start_aneg`
*   If you change settings, call `phy_start_aneg(phydev)` to restart auto-negotiation.
*   The PHY Lib handles the rest.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is RGMII?
    *   **A:** Reduced Gigabit Media Independent Interface. Uses DDR (Double Data Rate) on the clock to reduce pin count compared to GMII.
2.  **Q:** Why `netif_carrier_on`?
    *   **A:** It tells the kernel stack "The cable is plugged in". Without this, the kernel might not even try to send packets (depending on configuration).

### Challenge Task
> **Task:** "The Fake PHY".
> *   Write a dummy MDIO read function.
> *   Return ID `0x00137400` (Intel LXT971) for Reg 2/3.
> *   Return Link Up status for Reg 1.
> *   Verify Linux loads the `lxt` driver (or `Generic PHY`).

---

## 📚 Further Reading & References
- [Kernel Documentation: networking/phy.rst](https://www.kernel.org/doc/html/latest/networking/phy.html)

---
