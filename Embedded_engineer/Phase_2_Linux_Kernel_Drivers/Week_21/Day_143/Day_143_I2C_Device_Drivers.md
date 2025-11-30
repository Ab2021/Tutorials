# Day 143: I2C Device Drivers - Real World Implementation
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
1.  **Develop** a complete I2C driver for a Temperature/Pressure sensor (e.g., BMP280).
2.  **Implement** `regmap` for simplified register access and caching.
3.  **Expose** sensor data to Userspace via `sysfs` attributes.
4.  **Handle** endianness conversions (Big Endian vs Little Endian data).
5.  **Debug** I2C communication issues using `ftrace` and protocol analyzers.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   (Optional) BMP280 or similar I2C sensor.
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 142 (I2C Basics).
    *   Day 135 (Platform Drivers).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sensor Driver Pattern
Most sensor drivers follow a standard pattern:
1.  **Probe:**
    *   Read "Chip ID" register to verify presence.
    *   Reset the chip.
    *   Configure sampling rate / resolution.
2.  **Runtime:**
    *   Read data registers (often multi-byte).
    *   Convert raw values to engineering units (degC, hPa).
    *   Return to user.

### 🔹 Part 2: Regmap (Register Map)
Writing `i2c_smbus_read_byte_data` everywhere is tedious.
*   **Regmap** abstracts the bus (I2C/SPI) behind a unified API.
*   **Features:**
    *   Caching (Don't read hardware if value hasn't changed).
    *   Bulk Read/Write.
    *   Debugfs support (Dump all registers easily).
    *   Endianness handling.

---

## 💻 Implementation: The BMP280 Driver (Simplified)

> **Instruction:** We will write a driver for the Bosch BMP280 (Temp/Pressure).
> *   **Address:** 0x76 or 0x77.
> *   **ID Register:** 0xD0 (Value 0x58).

### 👨‍💻 Code Implementation

#### Step 1: Header Includes
```c
#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/regmap.h>
#include <linux/init.h>

#define BMP280_REG_ID       0xD0
#define BMP280_ID_VAL       0x58
#define BMP280_REG_CTRL_MEAS 0xF4
#define BMP280_REG_TEMP_MSB 0xFA
```

#### Step 2: Regmap Configuration
```c
static const struct regmap_config bmp280_regmap_config = {
    .reg_bits = 8,
    .val_bits = 8,
    .max_register = 0xFF,
    .cache_type = REGCACHE_RBTREE, // Enable caching
};
```

#### Step 3: Private Data
```c
struct bmp280_data {
    struct i2c_client *client;
    struct regmap *regmap;
    struct mutex lock;
};
```

#### Step 4: Probe Function
```c
static int bmp280_probe(struct i2c_client *client, const struct i2c_device_id *id) {
    struct bmp280_data *data;
    unsigned int chip_id;
    int ret;

    data = devm_kzalloc(&client->dev, sizeof(*data), GFP_KERNEL);
    if (!data) return -ENOMEM;

    data->client = client;
    i2c_set_clientdata(client, data); // Save private data

    // Init Regmap
    data->regmap = devm_regmap_init_i2c(client, &bmp280_regmap_config);
    if (IS_ERR(data->regmap)) {
        return PTR_ERR(data->regmap);
    }

    // Verify Chip ID
    ret = regmap_read(data->regmap, BMP280_REG_ID, &chip_id);
    if (ret < 0) return ret;

    if (chip_id != BMP280_ID_VAL) {
        dev_err(&client->dev, "Invalid Chip ID: 0x%x\n", chip_id);
        return -ENODEV;
    }

    // Configure (Normal Mode, Oversampling)
    // Write 0x27 to CTRL_MEAS (Temp x1, Press x1, Normal Mode)
    regmap_write(data->regmap, BMP280_REG_CTRL_MEAS, 0x27);

    dev_info(&client->dev, "BMP280 Probed Successfully!\n");
    return 0;
}
```

#### Step 5: Reading Temperature (Sysfs)
```c
static ssize_t temp_show(struct device *dev, struct device_attribute *attr, char *buf) {
    struct i2c_client *client = to_i2c_client(dev);
    struct bmp280_data *data = i2c_get_clientdata(client);
    u8 raw[3];
    s32 adc_T;
    int ret;

    // Bulk Read 3 bytes (MSB, LSB, XLSB)
    ret = regmap_bulk_read(data->regmap, BMP280_REG_TEMP_MSB, raw, 3);
    if (ret < 0) return ret;

    // Combine (20-bit value)
    adc_T = (raw[0] << 12) | (raw[1] << 4) | (raw[2] >> 4);

    // Note: Real driver needs compensation formula here. 
    // We just return raw ADC for simplicity.
    return sprintf(buf, "%d\n", adc_T);
}
static DEVICE_ATTR_RO(temp);

// Add to Probe:
// device_create_file(&client->dev, &dev_attr_temp);
```

---

## 💻 Implementation: EEPROM Driver (AT24C256)

> **Instruction:** EEPROMs are different. They act like storage. We use the `nvmem` subsystem usually, but let's write a raw character driver for learning.

### 👨‍💻 Code Implementation

#### Addressing Logic
AT24C256 uses 16-bit addresses.
*   Write: `[Addr High] [Addr Low] [Data]`
*   Read: Write `[Addr High] [Addr Low]`, then Read `[Data]`.

```c
static ssize_t eeprom_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos) {
    struct i2c_client *client = filp->private_data;
    u8 addr[2];
    u8 *tmp;
    struct i2c_msg msg[2];
    int ret;

    if (*f_pos > 32768) return 0; // EOF

    // 1. Set Address Pointer
    addr[0] = (*f_pos >> 8) & 0xFF;
    addr[1] = *f_pos & 0xFF;

    msg[0].addr = client->addr;
    msg[0].flags = 0; // Write
    msg[0].len = 2;
    msg[0].buf = addr;

    // 2. Read Data
    tmp = kmalloc(count, GFP_KERNEL);
    msg[1].addr = client->addr;
    msg[1].flags = I2C_M_RD;
    msg[1].len = count;
    msg[1].buf = tmp;

    ret = i2c_transfer(client->adapter, msg, 2);
    if (ret < 0) {
        kfree(tmp);
        return ret;
    }

    if (copy_to_user(buf, tmp, count)) {
        kfree(tmp);
        return -EFAULT;
    }

    *f_pos += count;
    kfree(tmp);
    return count;
}
```

---

## 🔬 Lab Exercise: Lab 143.1 - Regmap Caching

### 1. Lab Objectives
- Enable Regmap Caching.
- Read a register twice.
- Observe via Logic Analyzer (or `ftrace`) that the second read does NOT generate bus traffic.

### 2. Step-by-Step Guide
1.  In `regmap_config`: `.cache_type = REGCACHE_RBTREE`.
2.  Read ID register.
3.  Read ID register again.
4.  **Verification:**
    ```bash
    cd /sys/kernel/debug/regmap/0-0076/
    cat registers
    cat access  # Shows Hit/Miss stats
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Interrupts from I2C Devices
- **Goal:** Handle "Data Ready" interrupt.
- **Task:**
    1.  DTS: `interrupts = <...>;`
    2.  Probe: `devm_request_threaded_irq(client->irq, NULL, my_handler, ...);`
    3.  Why threaded? Because `regmap_read` sleeps (I2C is slow). You can't call it from Hard IRQ.

### Lab 3: Endianness
- **Goal:** Handle Big Endian sensor on Little Endian CPU.
- **Task:**
    1.  Use `be16_to_cpu()` when reading 16-bit registers manually.
    2.  Or configure `regmap_config.val_format_endian = REGMAP_ENDIAN_BIG`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Invalid Chip ID"
*   **Cause:** Wrong address (0x76 vs 0x77).
*   **Cause:** Device in reset.
*   **Cause:** SPI mode selected (some sensors support both, check CS pin).

#### 2. Data Corruption
*   **Cause:** Reading multi-byte registers one by one.
*   **Fix:** Use `regmap_bulk_read`. Most sensors latch the LSB when MSB is read (or vice versa) to ensure atomicity.

---

## ⚡ Optimization & Best Practices

### Block Reads
*   Always prefer reading X bytes in one transaction vs X transactions of 1 byte.
*   I2C overhead (Start/Addr/Ack) is significant.

### Power Management
*   Implement `runtime_suspend` and `runtime_resume`.
*   Put sensor to sleep when file is closed or after timeout.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we use `devm_regmap_init_i2c`?
    *   **A:** It handles memory allocation for the regmap structure and automatically frees it when the driver is removed.
2.  **Q:** Can I use `regmap` for a device that doesn't support I2C/SPI (e.g., Memory Mapped)?
    *   **A:** Yes! `devm_regmap_init_mmio`. It provides a consistent API regardless of the underlying bus.

### Challenge Task
> **Task:** "The Universal Sensor".
> *   Write a driver that supports BOTH I2C and SPI interfaces for the same sensor (e.g., BMP280).
> *   Create `bmp280_core.c` (Logic), `bmp280_i2c.c` (I2C Glue), `bmp280_spi.c` (SPI Glue).
> *   Use Regmap to hide the bus differences.

---

## 📚 Further Reading & References
- [Kernel Documentation: devicetree/bindings/i2c/i2c.txt](https://www.kernel.org/doc/Documentation/devicetree/bindings/i2c/i2c.txt)
- [Regmap API](https://www.kernel.org/doc/html/latest/driver-api/regmap.html)

---
