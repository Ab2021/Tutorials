# Day 145: SPI Device Drivers - Real World Implementation
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
1.  **Develop** a driver for an SPI Flash Memory (W25Qxx).
2.  **Integrate** with the MTD (Memory Technology Device) subsystem.
3.  **Implement** a driver for an SPI Display (ILI9341) using `fbtft` or direct SPI.
4.  **Optimize** throughput using `spi_write_then_read` and DMA-safe buffers.
5.  **Debug** timing issues with high-speed SPI devices.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   (Optional) SPI Flash chip or SPI TFT Display.
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 144 (SPI Basics).
    *   Flash Memory concepts (Pages, Sectors, Erasure).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The MTD Subsystem
Linux doesn't treat Flash memory like a normal Block Device (Hard Drive) because:
*   **Erasure:** You must erase a block (set to 1s) before writing (set to 0s).
*   **Wear:** Flash wears out.
*   **MTD:** The subsystem that handles raw flash. It sits below filesystems like JFFS2 or UBIFS.

### 🔹 Part 2: SPI Displays
*   **Command/Data (D/C) Pin:** Most displays have an extra GPIO to distinguish Command vs Pixel Data.
*   **Framebuffer:** The driver usually allocates a framebuffer in RAM and pushes it to the display over SPI periodically or on update.

---

## 💻 Implementation: SPI Flash Driver (Simplified)

> **Instruction:** We will write a driver that identifies the chip and allows reading raw bytes.

### 👨‍💻 Code Implementation

#### Step 1: Probe & ID Read

```c
#include <linux/module.h>
#include <linux/spi/spi.h>

#define CMD_READ_JEDEC_ID 0x9F

static int flash_probe(struct spi_device *spi) {
    u8 *tx_buf;
    u8 *rx_buf;
    int ret;

    // Allocate DMA-safe buffers
    tx_buf = devm_kzalloc(&spi->dev, 4, GFP_KERNEL);
    rx_buf = devm_kzalloc(&spi->dev, 4, GFP_KERNEL);
    if (!tx_buf || !rx_buf) return -ENOMEM;

    // Send Command
    tx_buf[0] = CMD_READ_JEDEC_ID;
    
    struct spi_transfer t = {
        .tx_buf = tx_buf,
        .rx_buf = rx_buf,
        .len = 4,
    };
    
    struct spi_message m;
    spi_message_init(&m);
    spi_message_add_tail(&t, &m);
    
    ret = spi_sync(spi, &m);
    if (ret) return ret;

    dev_info(&spi->dev, "Flash ID: Mfg=0x%02x, MemType=0x%02x, Cap=0x%02x\n",
             rx_buf[1], rx_buf[2], rx_buf[3]);
             
    return 0;
}
```

#### Step 2: Reading Data
```c
static ssize_t flash_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos) {
    struct spi_device *spi = filp->private_data;
    u8 cmd[4];
    struct spi_transfer t[2];
    struct spi_message m;
    u8 *kbuf;
    int ret;

    if (*f_pos > 0x100000) return 0; // 1MB limit

    // 1. Command Phase (0x03 + 24-bit Addr)
    cmd[0] = 0x03; // READ
    cmd[1] = (*f_pos >> 16) & 0xFF;
    cmd[2] = (*f_pos >> 8) & 0xFF;
    cmd[3] = *f_pos & 0xFF;

    memset(t, 0, sizeof(t));
    t[0].tx_buf = cmd;
    t[0].len = 4;
    
    // 2. Data Phase
    kbuf = kzalloc(count, GFP_KERNEL);
    t[1].rx_buf = kbuf;
    t[1].len = count;

    spi_message_init(&m);
    spi_message_add_tail(&t[0], &m);
    spi_message_add_tail(&t[1], &m);

    ret = spi_sync(spi, &m);
    if (ret) {
        kfree(kbuf);
        return ret;
    }

    if (copy_to_user(buf, kbuf, count)) {
        kfree(kbuf);
        return -EFAULT;
    }

    *f_pos += count;
    kfree(kbuf);
    return count;
}
```

---

## 💻 Implementation: SPI Display Driver (Concept)

> **Instruction:** Handling the D/C GPIO pin.

### 👨‍💻 Code Implementation

```c
struct display_data {
    struct spi_device *spi;
    struct gpio_desc *dc_gpio;
    struct gpio_desc *reset_gpio;
};

static void write_command(struct display_data *d, u8 cmd) {
    gpiod_set_value(d->dc_gpio, 0); // Command Mode
    spi_write(d->spi, &cmd, 1);
}

static void write_data(struct display_data *d, u8 *data, size_t len) {
    gpiod_set_value(d->dc_gpio, 1); // Data Mode
    spi_write(d->spi, data, len);
}

static int display_init(struct display_data *d) {
    // Hardware Reset
    gpiod_set_value(d->reset_gpio, 1);
    msleep(50);
    gpiod_set_value(d->reset_gpio, 0);
    msleep(50);
    gpiod_set_value(d->reset_gpio, 1);
    msleep(150);

    // Init Sequence (Example for ILI9341)
    write_command(d, 0x01); // SWRESET
    msleep(150);
    write_command(d, 0x11); // SLPOUT
    msleep(150);
    write_command(d, 0x29); // DISPON
    
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 145.1 - MTD Integration

### 1. Lab Objectives
- Instead of raw char driver, register as MTD device.
- Use `mtd_device_register`.

### 2. Step-by-Step Guide
1.  Include `<linux/mtd/mtd.h>`.
2.  Allocate `struct mtd_info`.
3.  Fill callbacks: `_read`, `_write`, `_erase`.
4.  Call `mtd_device_register(mtd, NULL, 0)`.
5.  **Verify:** `cat /proc/mtd`. You should see your device.

---

## 🧪 Additional / Advanced Labs

### Lab 2: SPI with Regmap
- **Goal:** Use Regmap for SPI.
- **Task:**
    1.  `devm_regmap_init_spi(spi, &config)`.
    2.  Config: `.reg_bits = 8`, `.val_bits = 8`, `.read_flag_mask = 0x80`.
    3.  Use `regmap_read/write`.

### Lab 3: Burst Write Optimization
- **Goal:** Optimize display updates.
- **Task:**
    1.  Instead of writing pixel-by-pixel, allocate a line buffer (e.g., 320 * 2 bytes).
    2.  Fill buffer.
    3.  Send entire buffer in one `spi_write` call.
    4.  Measure FPS improvement.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Flash ID is 0x000000 or 0xFFFFFF"
*   **Cause:** Wiring issue. MISO disconnected (0xFF) or shorted to GND (0x00).
*   **Cause:** CS timing. Some devices need CS to toggle between bytes (rare, but possible).

#### 2. Display is White/Black
*   **Cause:** Initialization sequence incorrect.
*   **Cause:** Backlight GPIO not enabled.

---

## ⚡ Optimization & Best Practices

### `spi_write_then_read`
*   Helper function for the common case: Write Command -> Read Data.
*   Uses internal DMA-safe buffer for small transfers (< 32 bytes).
*   `spi_write_then_read(spi, &cmd, 1, &data, 3);`

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need `devm_kzalloc` for SPI buffers?
    *   **A:** `devm_` is for automatic freeing. `kzalloc` (heap) is required for DMA. Stack memory (`u8 buf[10]`) cannot be used for DMA because it's not physically contiguous or cache-coherent in the way DMA expects.
2.  **Q:** What is the "D/C" pin?
    *   **A:** Data/Command. Used in displays (MIPI DBI Type C) to tell the controller if the incoming byte is a register address or pixel data.

### Challenge Task
> **Task:** "The Secure Flash".
> *   Implement the "Block Protect" feature of SPI Flash.
> *   Add an IOCTL to lock/unlock specific sectors.
> *   Read the Status Register (0x05) to verify.

---

## 📚 Further Reading & References
- [Kernel Documentation: mtd/spi-nor.rst](https://www.kernel.org/doc/html/latest/mtd/spi-nor.html)
- [FBTFT Driver Source](https://github.com/notro/fbtft)

---
