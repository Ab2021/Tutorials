# Day 144: SPI Subsystem Architecture
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
1.  **Compare** SPI vs I2C (Full Duplex, Chip Select, Speed).
2.  **Explain** the Linux SPI Subsystem (Master, Controller, Device, Driver).
3.  **Construct** SPI Transfers using `spi_message` and `spi_transfer`.
4.  **Configure** SPI Modes (CPOL, CPHA) and Max Speed in Device Tree.
5.  **Use** `spidev_test` to validate SPI loopback.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
    *   (Optional) Raspberry Pi with SPI loopback (MISO connected to MOSI).
*   **Software Required:**
    *   `spi-tools` (optional).
    *   Kernel Source.
*   **Prior Knowledge:**
    *   SPI Protocol (MISO, MOSI, SCK, CS).
    *   Day 142 (I2C Subsystem).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SPI vs I2C
| Feature | I2C | SPI |
| :--- | :--- | :--- |
| **Wires** | 2 (SDA, SCL) | 4 (MOSI, MISO, SCK, CS) |
| **Speed** | Slow (100k - 3.4M) | Fast (10M - 100M+) |
| **Duplex** | Half | Full (Simultaneous Tx/Rx) |
| **Addressing** | In-band (7-bit) | Out-of-band (Chip Select) |
| **Flow Control** | Clock Stretching | None (Usually) |

### 🔹 Part 2: Linux SPI Architecture
Similar to I2C, but with different terminology.
1.  **SPI Controller (`spi_controller` / `spi_master`):** The hardware block.
2.  **SPI Device (`spi_device`):** The slave peripheral.
3.  **SPI Driver (`spi_driver`):** The software for the slave.

### 🔹 Part 3: Data Structures
*   **`struct spi_transfer`:** Represents a single read/write chunk.
    *   `tx_buf`: Pointer to data to send.
    *   `rx_buf`: Pointer to buffer for received data.
    *   `len`: Length.
    *   `cs_change`: Toggle CS after this transfer?
*   **`struct spi_message`:** A list of transfers to be executed atomically.
    *   CS goes Low -> Transfer 1 -> Transfer 2 -> ... -> CS goes High.

---

## 💻 Implementation: SPI Loopback Test

> **Instruction:** Before writing a driver, verify the bus works using the generic `spidev` driver.

### 👨‍💻 Command Line Steps

#### Step 1: Enable spidev in DTS
In QEMU or RPi, ensure a node exists:
```dts
&spi0 {
    status = "okay";
    spidev@0 {
        compatible = "rohm,dh2228fv"; /* Generic compatible often used for spidev */
        reg = <0>; /* CS 0 */
        spi-max-frequency = <1000000>;
    };
};
```

#### Step 2: Run Loopback Test
Compile `tools/spi/spidev_test.c` from kernel source.
```bash
gcc -o spidev_test spidev_test.c
./spidev_test -D /dev/spidev0.0 -v
```
**Expected Output:**
```text
TX | FF FF FF FF FF FF
RX | FF FF FF FF FF FF
```
(If MISO/MOSI connected, RX matches TX).

---

## 💻 Implementation: The Dummy SPI Driver

> **Instruction:** Write a kernel driver that probes an SPI device.

### 👨‍💻 Code Implementation

#### Step 1: Driver Source (`dummy_spi.c`)

```c
#include <linux/module.h>
#include <linux/spi/spi.h>
#include <linux/init.h>

static int dummy_probe(struct spi_device *spi) {
    printk(KERN_INFO "Dummy SPI: Probed! CS=%d, MaxSpeed=%dHz\n", 
           spi->chip_select, spi->max_speed_hz);
    
    // Setup SPI Mode (Optional, usually done by core based on DTS)
    spi->mode = SPI_MODE_0;
    spi->bits_per_word = 8;
    return spi_setup(spi);
}

static void dummy_remove(struct spi_device *spi) {
    printk(KERN_INFO "Dummy SPI: Removed\n");
}

static const struct of_device_id dummy_dt_ids[] = {
    { .compatible = "org,dummy-spi" },
    { }
};
MODULE_DEVICE_TABLE(of, dummy_dt_ids);

static struct spi_driver dummy_driver = {
    .driver = {
        .name = "dummy_spi",
        .of_match_table = dummy_dt_ids,
    },
    .probe = dummy_probe,
    .remove = dummy_remove,
};

module_spi_driver(dummy_driver);
MODULE_LICENSE("GPL");
```

#### Step 2: DTS Overlay
```dts
&spi0 {
    status = "okay";
    my_dummy@0 {
        compatible = "org,dummy-spi";
        reg = <0>; /* Chip Select 0 */
        spi-max-frequency = <500000>; /* 500 kHz */
        spi-cpol; /* Optional: Clock Polarity 1 */
        spi-cpha; /* Optional: Clock Phase 1 */
    };
};
```

---

## 💻 Implementation: Synchronous Transfer

> **Instruction:** Send "Hello" and print what comes back.

### 👨‍💻 Code Implementation

```c
static int spi_send_hello(struct spi_device *spi) {
    char tx_buf[] = "Hello";
    char rx_buf[6];
    int ret;
    
    // 1. Define Transfer
    struct spi_transfer t = {
        .tx_buf = tx_buf,
        .rx_buf = rx_buf,
        .len = sizeof(tx_buf),
    };
    
    // 2. Define Message
    struct spi_message m;
    spi_message_init(&m);
    spi_message_add_tail(&t, &m);
    
    // 3. Execute (Synchronous - Sleeps)
    ret = spi_sync(spi, &m);
    if (ret) {
        printk("SPI Transfer failed: %d\n", ret);
        return ret;
    }
    
    printk("SPI RX: %s\n", rx_buf);
    return 0;
}
```

---

## 💻 Implementation: Asynchronous Transfer

> **Instruction:** Don't block. Use a callback.

### 👨‍💻 Code Implementation

```c
static void spi_complete(void *arg) {
    printk("SPI Async Complete!\n");
}

static int spi_send_async(struct spi_device *spi) {
    // Buffers MUST be DMA-safe (kmalloc'd, not on stack!)
    char *tx_buf = kmalloc(6, GFP_KERNEL);
    char *rx_buf = kmalloc(6, GFP_KERNEL);
    struct spi_transfer *t = kzalloc(sizeof(*t), GFP_KERNEL);
    struct spi_message *m = kzalloc(sizeof(*m), GFP_KERNEL);
    
    strcpy(tx_buf, "Async");
    
    t->tx_buf = tx_buf;
    t->rx_buf = rx_buf;
    t->len = 6;
    
    spi_message_init(m);
    spi_message_add_tail(t, m);
    m->complete = spi_complete;
    m->context = NULL;
    
    return spi_async(spi, m);
}
```
*Note: Memory management for async is tricky. You must free buffers in the callback.*

---

## 🔬 Lab Exercise: Lab 144.1 - Multi-Transfer Message

### 1. Lab Objectives
- Simulate a protocol: Command (1 byte) + Data (3 bytes).
- Keep CS Low during the whole transaction.

### 2. Step-by-Step Guide
1.  Create `struct spi_transfer t[2]`.
2.  `t[0].tx_buf = &cmd; t[0].len = 1; t[0].cs_change = 0;`
3.  `t[1].tx_buf = data; t[1].len = 3;`
4.  Add both to message.
5.  `spi_sync`.
6.  **Verify:** Use Logic Analyzer. CS should stay Low between byte 1 and byte 2.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bit-Banging SPI (`spi-gpio`)
- **Goal:** Use GPIOs as SPI.
- **Task:**
    1.  DTS: Define `spi-gpio` node with `sck-gpios`, `miso-gpios`, `mosi-gpios`.
    2.  Bind your driver to it.
    3.  It works exactly the same! The kernel handles the bit-banging.

### Lab 3: Chip Select High
- **Goal:** Handle active-high CS.
- **Task:**
    1.  DTS: `reg = <0>; spi-cs-high;`
    2.  Verify CS line goes High during transfer.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "SPI transfer failed"
*   **Cause:** DMA mapping failed (Stack memory used for buffer?).
*   **Fix:** Always use `kmalloc` for buffers, even for synchronous transfers, to be safe with DMA-enabled controllers.

#### 2. Garbage Data
*   **Cause:** Mode Mismatch (CPOL/CPHA).
*   **Fix:** Check datasheet. Mode 0 (0,0) and Mode 3 (1,1) are most common.

---

## ⚡ Optimization & Best Practices

### DMA
*   SPI is fast. Interrupt overhead is high.
*   Ensure the SPI Controller driver enables DMA.
*   For small transfers (< 32 bytes), PIO (Polling/IRQ) might be faster than setting up DMA.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `cs_change`?
    *   **A:** It controls the CS line *between* transfers in a message. Usually 0 (keep CS active). If 1, CS toggles inactive then active again before the next transfer.
2.  **Q:** Can I have multiple SPI devices?
    *   **A:** Yes, they share SCK/MOSI/MISO but have separate CS lines (CS0, CS1...).

### Challenge Task
> **Task:** "The SPI Flash ID Reader".
> *   Connect an SPI Flash (e.g., W25Qxx).
> *   Send Command `0x9F` (JEDEC ID).
> *   Read 3 bytes.
> *   Print Manufacturer and Device ID.

---

## 📚 Further Reading & References
- [Kernel Documentation: spi/spi-summary.rst](https://www.kernel.org/doc/html/latest/spi/spi-summary.html)
- [SPI Protocol Wikipedia](https://en.wikipedia.org/wiki/Serial_Peripheral_Interface)

---
