# Day 142: I2C Subsystem Architecture
## Phase 2: Linux Kernel & Device Drivers | Week 21: I2C and SPI Drivers

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
1.  **Explain** the Linux I2C Subsystem architecture (Core, Algorithm, Adapter, Client, Driver).
2.  **Distinguish** between an I2C Adapter Driver (Controller) and an I2C Client Driver (Peripheral).
3.  **Navigate** the I2C Core API (`i2c_transfer`, `i2c_master_send`, `i2c_master_recv`).
4.  **Instantiate** I2C devices via Device Tree and Userspace (`new_device`).
5.  **Use** `i2c-tools` (`i2cdetect`, `i2cdump`) to debug the bus.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   (Optional) Raspberry Pi with an I2C sensor (e.g., BMP280, MPU6050).
*   **Software Required:**
    *   `i2c-tools` (`sudo apt install i2c-tools`).
    *   Kernel Source.
*   **Prior Knowledge:**
    *   I2C Protocol (Start, Stop, ACK, NACK, Address).
    *   Day 135 (Platform Drivers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Layered Architecture
The Linux I2C subsystem is designed to be hardware-agnostic.
1.  **I2C Core (`drivers/i2c/i2c-core.c`):** The middleman. Manages registration of adapters and drivers. Provides the API.
2.  **I2C Algorithm (`i2c_algorithm`):** Defines *how* to talk. (e.g., bit-banging vs hardware controller).
3.  **I2C Adapter (`i2c_adapter`):** Represents the physical controller (The "Master"). Example: `i2c-0` (The SoC's I2C1 block).
4.  **I2C Client (`i2c_client`):** Represents a device connected to the bus (The "Slave"). Example: An EEPROM at address 0x50.
5.  **I2C Driver (`i2c_driver`):** The software that knows how to talk to the Client.

### 🔹 Part 2: Adapter vs Client
*   **Adapter Driver:** You write this if you are building a new CPU/SoC. It talks to the I2C registers of the CPU.
*   **Client Driver:** You write this if you are connecting a sensor (accelerometer, temp sensor) to the board. **This is what 99% of embedded engineers do.**

### 🔹 Part 3: Communication Model
*   **Master Transfer:** The CPU initiates everything.
*   **Messages (`struct i2c_msg`):** The atomic unit of I2C I/O.
    *   `addr`: Slave address.
    *   `flags`: Read/Write.
    *   `len`: Length.
    *   `buf`: Data pointer.

---

## 💻 Implementation: Exploring I2C from Userspace

> **Instruction:** Before writing kernel code, let's understand what the kernel exposes to userspace.

### 👨‍💻 Command Line Steps

#### Step 1: List Adapters
```bash
i2cdetect -l
# Output example:
# i2c-0  i2c       Synopsys DesignWare I2C adapter    I2C adapter
# i2c-1  i2c       bcm2835 (i2c@7e804000)             I2C adapter
```

#### Step 2: Scan the Bus
Scan bus 1 for devices.
```bash
i2cdetect -y 1
# Output grid:
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
# 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# 50: 50 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# 60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- -- 
```
*   `50`: A device is present at 0x50 (likely EEPROM).
*   `UU`: A driver is already managing this device (Kernel has claimed it).
*   `--`: No response.

#### Step 3: Dump Registers
Read all registers from device 0x50 on bus 1.
```bash
i2cdump -y 1 0x50
```

---

## 💻 Implementation: The Dummy I2C Client Driver

> **Instruction:** We will write a simple client driver that probes when it finds a device at a specific address.

### 👨‍💻 Code Implementation

#### Step 1: Driver Source (`dummy_i2c.c`)

```c
#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/init.h>

// 1. Probe Function
static int dummy_probe(struct i2c_client *client, const struct i2c_device_id *id) {
    printk(KERN_INFO "Dummy I2C: Probed! Addr=0x%x\n", client->addr);
    
    // Check functionality (Optional but good practice)
    if (!i2c_check_functionality(client->adapter, I2C_FUNC_I2C)) {
        return -EIO;
    }
    
    return 0;
}

// 2. Remove Function
static void dummy_remove(struct i2c_client *client) {
    printk(KERN_INFO "Dummy I2C: Removed\n");
}

// 3. Device ID Table (Legacy)
static const struct i2c_device_id dummy_id[] = {
    { "dummy_device", 0 },
    { }
};
MODULE_DEVICE_TABLE(i2c, dummy_id);

// 4. Device Tree Match Table (Modern)
static const struct of_device_id dummy_dt_ids[] = {
    { .compatible = "org,dummy-i2c" },
    { }
};
MODULE_DEVICE_TABLE(of, dummy_dt_ids);

// 5. Driver Structure
static struct i2c_driver dummy_driver = {
    .driver = {
        .name = "dummy_i2c",
        .of_match_table = dummy_dt_ids,
    },
    .probe = dummy_probe,
    .remove = dummy_remove,
    .id_table = dummy_id,
};

module_i2c_driver(dummy_driver); // Helper macro
MODULE_LICENSE("GPL");
```

#### Step 2: Instantiating the Device
Since we don't have a real sensor in QEMU (unless configured), we can simulate one or use the "new_device" interface.

**Method A: Userspace Instantiation**
1.  Load the driver: `insmod dummy_i2c.ko`.
2.  Tell the adapter to create a device:
    ```bash
    echo "dummy_device 0x55" > /sys/bus/i2c/devices/i2c-0/new_device
    ```
3.  **Result:** `dmesg` shows "Dummy I2C: Probed! Addr=0x55".

**Method B: Device Tree (QEMU Overlay)**
```dts
&i2c0 {
    status = "okay";
    my_dummy@55 {
        compatible = "org,dummy-i2c";
        reg = <0x55>;
    };
};
```

---

## 💻 Implementation: I2C Communication

> **Instruction:** How to read/write data in the driver.

### 👨‍💻 Code Implementation

#### Reading a Register (8-bit)
```c
static int i2c_read_reg(struct i2c_client *client, u8 reg) {
    int ret;
    u8 val;
    
    // Write Register Address
    struct i2c_msg msg[2];
    
    // Msg 1: Write Address
    msg[0].addr = client->addr;
    msg[0].flags = 0; // Write
    msg[0].len = 1;
    msg[0].buf = &reg;
    
    // Msg 2: Read Value
    msg[1].addr = client->addr;
    msg[1].flags = I2C_M_RD; // Read
    msg[1].len = 1;
    msg[1].buf = &val;
    
    ret = i2c_transfer(client->adapter, msg, 2);
    if (ret < 0) return ret;
    
    return val;
}
```

#### Using SMBus API (Easier)
Most devices follow the SMBus standard (Register-based access).
```c
// Read Byte from Register
s32 val = i2c_smbus_read_byte_data(client, 0x10);

// Write Byte to Register
i2c_smbus_write_byte_data(client, 0x10, 0xAB);
```
*Note: Prefer SMBus API if possible. It handles the `i2c_msg` construction for you.*

---

## 🔬 Lab Exercise: Lab 142.1 - The I2C Scanner Driver

### 1. Lab Objectives
- Write a kernel module that iterates 0x03 to 0x77.
- Tries to read 1 byte from each address.
- Prints detected addresses to dmesg.

### 2. Step-by-Step Guide
1.  Get the adapter: `struct i2c_adapter *adap = i2c_get_adapter(0);`
2.  Loop `addr` from 0x03 to 0x77.
3.  Create a dummy client:
    ```c
    struct i2c_client client = {
        .adapter = adap,
        .addr = addr
    };
    ```
4.  Try `i2c_smbus_read_byte(&client)`. If `>= 0`, device exists.
5.  `i2c_put_adapter(adap);`

---

## 🧪 Additional / Advanced Labs

### Lab 2: 10-bit Addressing
- **Goal:** Support extended addressing.
- **Task:**
    1.  Set `client->flags |= I2C_CLIENT_TEN`.
    2.  Try to communicate with a 10-bit device (rare, but good for theory).

### Lab 3: I2C Muxing
- **Goal:** Understand `i2c-mux`.
- **Scenario:** You have 4 sensors with the *same* address (0x50).
- **Solution:** Use an I2C Mux (TCA9548A).
- **Task:** Review the DTS for an I2C mux. It creates virtual adapters (`i2c-2`, `i2c-3`...) for each channel.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Transfer failed" (-EREMOTEIO)
*   **Cause:** NACK received. No device at that address.
*   **Fix:** Check wiring (SDA/SCL swapped?). Check power.

#### 2. "Arbitration Lost" (-EAGAIN)
*   **Cause:** Multi-master collision.
*   **Fix:** Rare in simple embedded, but check if another master is driving the bus.

#### 3. "Controller timed out"
*   **Cause:** Clock stretching. The slave is holding SCL low for too long.
*   **Cause:** Missing Pull-up resistors! (SDA/SCL are Open Drain).

---

## ⚡ Optimization & Best Practices

### DMA
*   For large transfers (e.g., loading firmware to a touch controller), ensure the I2C Adapter driver supports DMA. The Client driver usually doesn't need to change (Core handles it).

### Atomic Context
*   `i2c_transfer` can sleep. **NEVER** call it from an ISR.
*   If you need I2C in an ISR, use a Threaded IRQ (Day 132).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `i2c_transfer` and `i2c_master_send`?
    *   **A:** `i2c_transfer` is the generic function taking an array of messages (Read/Write mixed). `i2c_master_send` is a wrapper for a single Write message.
2.  **Q:** Why do we need Pull-up resistors?
    *   **A:** I2C is Open Drain. Devices only drive Low. The resistor pulls the line High when idle. Without it, the line floats.

### Challenge Task
> **Task:** "The EEPROM Dumper".
> *   Write a driver for an AT24C256 EEPROM.
> *   Implement `read()` file operation.
> *   When user reads `/dev/eeprom`, read 256 bytes from the device using `i2c_transfer`.

---

## 📚 Further Reading & References
- [Kernel Documentation: i2c/writing-clients.rst](https://www.kernel.org/doc/html/latest/i2c/writing-clients.html)
- [Kernel Documentation: i2c/instantiating-devices.rst](https://www.kernel.org/doc/html/latest/i2c/instantiating-devices.html)

---
