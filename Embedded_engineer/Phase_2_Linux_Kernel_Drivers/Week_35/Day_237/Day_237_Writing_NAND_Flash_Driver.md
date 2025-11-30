# Day 237: Writing a NAND Flash Driver
## Phase 2: Linux Kernel & Device Drivers | Week 35: Advanced Storage & Filesystems

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
1.  **Understand** NAND flash controller architecture and timing requirements.
2.  **Implement** a NAND chip driver using the MTD NAND framework.
3.  **Configure** ECC (Error Correction Code) engines.
4.  **Handle** bad block management and marking.
5.  **Optimize** NAND operations for performance.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   Linux PC.
    *   Optional: Development board with NAND flash.
*   **Software Required:**
    *   Kernel Source with MTD and NAND support.
    *   Logic analyzer or oscilloscope (for hardware debugging).
*   **Prior Knowledge:**
    *   Day 236 (MTD Subsystem).
    *   Basic digital electronics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: NAND Flash Architecture

#### Physical Organization

```
NAND Chip
├── LUN (Logical Unit) 0
│   ├── Plane 0
│   │   ├── Block 0 (Erase Unit)
│   │   │   ├── Page 0 (Write Unit: 2KB + 64B OOB)
│   │   │   ├── Page 1
│   │   │   └── ... (64 pages typical)
│   │   ├── Block 1
│   │   └── ... (1024-4096 blocks)
│   └── Plane 1
└── LUN 1
```

#### Key Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| **Page Size** | 2KB, 4KB, 8KB | Minimum read/write unit |
| **OOB Size** | 64B, 128B, 256B | Spare area for metadata/ECC |
| **Block Size** | 128KB, 256KB, 512KB | Minimum erase unit |
| **Blocks per LUN** | 1024-4096 | Total blocks |
| **Erase Cycles** | 10K (MLC), 100K (SLC) | Endurance |

### 🔹 Part 2: NAND Commands

NAND flash uses a command-based interface:

| Command | Code | Description |
|---------|------|-------------|
| **READ** | 0x00, 0x30 | Read page |
| **READ ID** | 0x90 | Read chip ID |
| **RESET** | 0xFF | Reset chip |
| **PROGRAM** | 0x80, 0x10 | Write page |
| **ERASE** | 0x60, 0xD0 | Erase block |
| **READ STATUS** | 0x70 | Get status register |
| **READ PARAMETER PAGE** | 0xEC | Get ONFI parameters |

### 🔹 Part 3: NAND Timing

Critical timing parameters (from datasheet):

```c
struct nand_sdr_timings {
    u32 tCLS_min;   // CLE setup time
    u32 tCLH_min;   // CLE hold time
    u32 tCS_min;    // CE setup time
    u32 tCH_min;    // CE hold time
    u32 tWP_min;    // WE pulse width
    u32 tALS_min;   // ALE setup time
    u32 tALH_min;   // ALE hold time
    u32 tDS_min;    // Data setup time
    u32 tDH_min;    // Data hold time
    u32 tWC_min;    // Write cycle time
    u32 tRC_min;    // Read cycle time
    u32 tREA_max;   // RE access time
    u32 tRHW_min;   // RE high to WE low
    u32 tWHR_min;   // WE high to RE low
    // ... many more
};
```

### 🔹 Part 4: ECC (Error Correction Code)

**Why ECC is Critical:**
- NAND cells degrade over time.
- Bit flips occur due to read disturb, program disturb, retention loss.
- Without ECC, data corruption is inevitable.

**ECC Types:**

1.  **Hamming Code:**
    *   Corrects 1 bit per 512 bytes.
    *   Simple, low overhead.
    *   Insufficient for modern NAND.

2.  **BCH (Bose-Chaudhuri-Hocquenghem):**
    *   Corrects 4, 8, 16, or more bits per page.
    *   Widely used.
    *   Hardware or software implementation.

3.  **LDPC (Low-Density Parity-Check):**
    *   Corrects many bits.
    *   Used in modern high-density NAND.
    *   Requires significant computation.

**ECC Placement:**
- Stored in OOB area.
- Calculated during write, verified during read.

---

## 💻 Implementation: NAND Controller Driver

> **Instruction:** Implement a simple NAND driver using the MTD NAND framework.

### 👨‍💻 Code Implementation

```c
#include <linux/module.h>
#include <linux/mtd/mtd.h>
#include <linux/mtd/rawnand.h>
#include <linux/mtd/partitions.h>
#include <linux/platform_device.h>
#include <linux/io.h>
#include <linux/iopoll.h>

#define NAND_BASE_ADDR  0x10000000  // Example memory-mapped address
#define NAND_CMD_OFFSET 0x00
#define NAND_ADDR_OFFSET 0x04
#define NAND_DATA_OFFSET 0x08
#define NAND_CTRL_OFFSET 0x0C

struct my_nand_controller {
    struct nand_controller controller;
    void __iomem *regs;
    struct completion complete;
};

struct my_nand_chip {
    struct nand_chip chip;
    struct my_nand_controller *controller;
};

// Hardware access functions
static void my_nand_cmd_ctrl(struct nand_chip *chip, int cmd,
                             unsigned int ctrl) {
    struct my_nand_chip *my_chip = container_of(chip, struct my_nand_chip, chip);
    struct my_nand_controller *my_ctrl = my_chip->controller;
    
    if (ctrl & NAND_CTRL_CHANGE) {
        if (ctrl & NAND_CLE)
            writeb(cmd, my_ctrl->regs + NAND_CMD_OFFSET);
        else if (ctrl & NAND_ALE)
            writeb(cmd, my_ctrl->regs + NAND_ADDR_OFFSET);
    }
    
    if (cmd != NAND_CMD_NONE)
        writeb(cmd, my_ctrl->regs + NAND_DATA_OFFSET);
}

static int my_nand_dev_ready(struct nand_chip *chip) {
    struct my_nand_chip *my_chip = container_of(chip, struct my_nand_chip, chip);
    struct my_nand_controller *my_ctrl = my_chip->controller;
    u32 status;
    
    status = readl(my_ctrl->regs + NAND_CTRL_OFFSET);
    return (status & 0x01) ? 1 : 0;  // Check R/B bit
}

static void my_nand_read_buf(struct nand_chip *chip, uint8_t *buf, int len) {
    struct my_nand_chip *my_chip = container_of(chip, struct my_nand_chip, chip);
    struct my_nand_controller *my_ctrl = my_chip->controller;
    int i;
    
    for (i = 0; i < len; i++)
        buf[i] = readb(my_ctrl->regs + NAND_DATA_OFFSET);
}

static void my_nand_write_buf(struct nand_chip *chip, const uint8_t *buf, int len) {
    struct my_nand_chip *my_chip = container_of(chip, struct my_nand_chip, chip);
    struct my_nand_controller *my_ctrl = my_chip->controller;
    int i;
    
    for (i = 0; i < len; i++)
        writeb(buf[i], my_ctrl->regs + NAND_DATA_OFFSET);
}

// ECC functions (software BCH example)
static int my_nand_ecc_calculate(struct nand_chip *chip,
                                 const uint8_t *dat, uint8_t *ecc_code) {
    // Use software BCH library
    struct mtd_info *mtd = nand_to_mtd(chip);
    struct nand_ecc_ctrl *ecc = &chip->ecc;
    
    // Calculate ECC for the data
    // This is a placeholder - real implementation would use bch_encode()
    memset(ecc_code, 0, ecc->bytes);
    
    return 0;
}

static int my_nand_ecc_correct(struct nand_chip *chip, uint8_t *dat,
                               uint8_t *read_ecc, uint8_t *calc_ecc) {
    struct nand_ecc_ctrl *ecc = &chip->ecc;
    unsigned int errloc[8];
    int i, count;
    
    // Compare ECCs and correct errors
    // This is a placeholder - real implementation would use bch_decode()
    
    // Return number of corrected bits, or -EBADMSG if uncorrectable
    return 0;
}

// Controller operations
static int my_nand_attach_chip(struct nand_chip *chip) {
    struct mtd_info *mtd = nand_to_mtd(chip);
    struct nand_ecc_ctrl *ecc = &chip->ecc;
    
    // Configure ECC
    ecc->mode = NAND_ECC_HW;  // or NAND_ECC_SOFT for software ECC
    ecc->size = 512;          // ECC step size
    ecc->bytes = 7;           // ECC bytes per step (BCH-4)
    ecc->strength = 4;        // Correct up to 4 bits
    ecc->calculate = my_nand_ecc_calculate;
    ecc->correct = my_nand_ecc_correct;
    ecc->hwctl = NULL;        // No hardware control needed
    
    // Set up timing
    // chip->setup_data_interface = my_nand_setup_data_interface;
    
    return 0;
}

static const struct nand_controller_ops my_nand_controller_ops = {
    .attach_chip = my_nand_attach_chip,
};

// Probe function
static int my_nand_probe(struct platform_device *pdev) {
    struct my_nand_controller *my_ctrl;
    struct my_nand_chip *my_chip;
    struct nand_chip *chip;
    struct mtd_info *mtd;
    struct resource *res;
    int ret;
    
    // Allocate controller
    my_ctrl = devm_kzalloc(&pdev->dev, sizeof(*my_ctrl), GFP_KERNEL);
    if (!my_ctrl)
        return -ENOMEM;
    
    // Map registers
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    my_ctrl->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(my_ctrl->regs))
        return PTR_ERR(my_ctrl->regs);
    
    // Initialize controller
    nand_controller_init(&my_ctrl->controller);
    my_ctrl->controller.ops = &my_nand_controller_ops;
    init_completion(&my_ctrl->complete);
    
    // Allocate chip
    my_chip = devm_kzalloc(&pdev->dev, sizeof(*my_chip), GFP_KERNEL);
    if (!my_chip)
        return -ENOMEM;
    
    my_chip->controller = my_ctrl;
    chip = &my_chip->chip;
    mtd = nand_to_mtd(chip);
    
    // Set up chip
    chip->controller = &my_ctrl->controller;
    chip->legacy.cmd_ctrl = my_nand_cmd_ctrl;
    chip->legacy.dev_ready = my_nand_dev_ready;
    chip->legacy.read_buf = my_nand_read_buf;
    chip->legacy.write_buf = my_nand_write_buf;
    chip->legacy.chip_delay = 20;  // Delay in microseconds
    
    // Set MTD name
    mtd->name = "my-nand";
    mtd->owner = THIS_MODULE;
    mtd->dev.parent = &pdev->dev;
    
    // Scan for NAND chip
    ret = nand_scan(chip, 1);  // 1 = number of chips
    if (ret)
        return ret;
    
    // Register MTD device
    ret = mtd_device_register(mtd, NULL, 0);
    if (ret) {
        nand_cleanup(chip);
        return ret;
    }
    
    platform_set_drvdata(pdev, my_ctrl);
    
    dev_info(&pdev->dev, "NAND: %lluMB, page size %d, block size %d\n",
             (unsigned long long)chip->chipsize >> 20,
             mtd->writesize, mtd->erasesize);
    
    return 0;
}

static int my_nand_remove(struct platform_device *pdev) {
    struct my_nand_controller *my_ctrl = platform_get_drvdata(pdev);
    struct my_nand_chip *my_chip = container_of(my_ctrl->controller.chips[0],
                                                 struct my_nand_chip, chip);
    struct nand_chip *chip = &my_chip->chip;
    struct mtd_info *mtd = nand_to_mtd(chip);
    
    mtd_device_unregister(mtd);
    nand_cleanup(chip);
    
    return 0;
}

static const struct of_device_id my_nand_dt_ids[] = {
    { .compatible = "vendor,my-nand" },
    { /* sentinel */ }
};
MODULE_DEVICE_TABLE(of, my_nand_dt_ids);

static struct platform_driver my_nand_driver = {
    .driver = {
        .name = "my-nand",
        .of_match_table = my_nand_dt_ids,
    },
    .probe = my_nand_probe,
    .remove = my_nand_remove,
};

module_platform_driver(my_nand_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Embedded Engineer");
MODULE_DESCRIPTION("Simple NAND Flash Controller Driver");
```

---

## 🔬 Lab Exercise: Lab 237.1 - NAND Chip Detection

### 1. Lab Objectives
- Detect NAND chip and read its ID.
- Verify chip parameters.

### 2. Step-by-Step Guide

1.  **Device Tree Entry:**
    ```dts
    nand@10000000 {
        compatible = "vendor,my-nand";
        reg = <0x10000000 0x1000>;
        #address-cells = <1>;
        #size-cells = <0>;
        
        nand-ecc-mode = "hw";
        nand-ecc-strength = <4>;
        nand-ecc-step-size = <512>;
    };
    ```

2.  **Load Driver:**
    ```bash
    insmod my_nand.ko
    ```

3.  **Check Detection:**
    ```bash
    dmesg | grep -i nand
    # Should show:
    # NAND device: Manufacturer ID: 0xec, Chip ID: 0xda
    # NAND: 256MB, page size 2048, block size 131072
    ```

4.  **Verify MTD:**
    ```bash
    cat /proc/mtd
    mtdinfo /dev/mtd0
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Bad Block Scanning
- **Goal:** Scan for factory-marked bad blocks.
- **Task:**
    ```bash
    # Use nand_bbt.c functions
    # Factory bad blocks have specific markers in OOB
    sudo nand-utils/nandtest -m /dev/mtd0
    ```

### Lab 3: ECC Testing
- **Goal:** Inject bit errors and verify ECC correction.
- **Task:**
    1.  Write known pattern.
    2.  Read raw (without ECC).
    3.  Flip bits manually.
    4.  Write back.
    5.  Read with ECC - should correct.

### Lab 4: Performance Optimization
- **Goal:** Optimize read/write performance.
- **Task:**
    1.  Implement DMA for data transfer.
    2.  Use multi-plane operations.
    3.  Enable cache programming.
    4.  Benchmark with `mtd_speedtest`.

### Lab 5: Wear Leveling Analysis
- **Goal:** Understand wear patterns.
- **Task:**
    1.  Write test pattern repeatedly to same blocks.
    2.  Monitor erase counts.
    3.  Observe when blocks become bad.
    4.  Calculate actual vs. specified endurance.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Chip Not Detected
*   **Symptoms:** `nand_scan` fails, no ID read.
*   **Causes:**
    *   Incorrect register mapping.
    *   Wrong CE (Chip Enable) signal.
    *   Timing violations.
*   **Debug:**
    ```c
    // Add debug in cmd_ctrl
    pr_info("CMD: 0x%02x, CTRL: 0x%02x\n", cmd, ctrl);
    
    // Manually send READ ID command
    chip->legacy.cmd_ctrl(chip, NAND_CMD_READID, NAND_CTRL_CLE);
    chip->legacy.cmd_ctrl(chip, 0x00, NAND_CTRL_ALE);
    // Read 5 bytes
    ```

#### 2. ECC Errors
*   **Symptoms:** Uncorrectable ECC errors on read.
*   **Causes:**
    *   Wrong ECC algorithm.
    *   ECC bytes stored in wrong OOB location.
    *   Insufficient ECC strength.
*   **Debug:**
    ```bash
    # Enable MTD debug
    echo 3 > /sys/module/mtd/parameters/debug
    # Read and check errors
    nanddump -o /dev/mtd0
    ```

#### 3. Timing Issues
*   **Symptoms:** Random failures, data corruption.
*   **Causes:**
    *   tWP, tRC violations.
    *   Clock too fast for NAND chip.
*   **Debug:**
    *   Use logic analyzer to verify timing.
    *   Add delays in `cmd_ctrl`.
    *   Implement proper `setup_data_interface`.

#### 4. Bad Block Handling
*   **Symptoms:** Erase/write fails on certain blocks.
*   **Causes:**
    *   Factory bad blocks not marked.
    *   Runtime bad blocks not detected.
*   **Fix:**
    ```c
    // Check before erase
    if (chip->block_bad(mtd, offs, 0)) {
        pr_warn("Skipping bad block at 0x%llx\n", offs);
        continue;
    }
    ```

---

## ⚡ Optimization & Best Practices

### 1. DMA Transfer
```c
static void my_nand_read_buf_dma(struct nand_chip *chip, uint8_t *buf, int len) {
    struct my_nand_controller *my_ctrl = ...;
    dma_addr_t dma_addr;
    
    dma_addr = dma_map_single(dev, buf, len, DMA_FROM_DEVICE);
    
    // Configure DMA controller
    writel(dma_addr, my_ctrl->regs + DMA_ADDR);
    writel(len, my_ctrl->regs + DMA_LEN);
    writel(DMA_START, my_ctrl->regs + DMA_CTRL);
    
    // Wait for completion
    wait_for_completion(&my_ctrl->complete);
    
    dma_unmap_single(dev, dma_addr, len, DMA_FROM_DEVICE);
}
```

### 2. Multi-Plane Operations
```c
// Read from two planes simultaneously
static int my_nand_multiplane_read(struct nand_chip *chip,
                                   int page1, int page2) {
    // Send commands to both planes
    chip->legacy.cmdfunc(chip, NAND_CMD_READ0, 0, page1);
    chip->legacy.cmdfunc(chip, NAND_CMD_READ0_MULTIPLANE, 0, page2);
    chip->legacy.cmdfunc(chip, NAND_CMD_READSTART, -1, -1);
    
    // Wait and read data
    chip->legacy.waitfunc(chip);
    // ... read data from both planes
}
```

### 3. Cache Programming
```c
// Write multiple pages faster
static int my_nand_cache_program(struct nand_chip *chip,
                                 const uint8_t *buf, int page) {
    chip->legacy.cmdfunc(chip, NAND_CMD_SEQIN, 0, page);
    chip->legacy.write_buf(chip, buf, mtd->writesize);
    chip->legacy.cmdfunc(chip, NAND_CMD_CACHEDPROG, -1, -1);
    
    // Don't wait - next page can be programmed immediately
    return 0;
}
```

### 4. On-Die ECC
```c
// Use NAND chip's internal ECC
static int my_nand_ondie_ecc_setup(struct nand_chip *chip) {
    // Enable on-die ECC feature
    uint8_t feature[4] = {0x08, 0x00, 0x00, 0x00};
    
    chip->legacy.cmdfunc(chip, NAND_CMD_SET_FEATURES, 0x90, -1);
    chip->legacy.write_buf(chip, feature, 4);
    
    // Configure MTD to use on-die ECC
    chip->ecc.mode = NAND_ECC_ON_DIE;
    
    return 0;
}
```

---

## 📊 Performance Benchmarking

### Test Suite
```bash
# Sequential read
mtd_speedtest dev=/dev/mtd0

# Random read/write
mtd_stresstest dev=/dev/mtd0

# Torture test
mtd_torturetest dev=/dev/mtd0 eb=100
```

### Expected Performance

| Operation | SLC NAND | MLC NAND | TLC NAND |
|-----------|----------|----------|----------|
| **Page Read** | 25 µs | 50 µs | 75 µs |
| **Page Program** | 200 µs | 600 µs | 1000 µs |
| **Block Erase** | 2 ms | 3 ms | 5 ms |
| **Sequential Read** | 40 MB/s | 30 MB/s | 20 MB/s |
| **Sequential Write** | 10 MB/s | 5 MB/s | 3 MB/s |

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why can't you write to a NAND page twice without erasing?
    *   **A:** NAND cells can only transition from 1→0 during program. To change 0→1, the entire block must be erased (all bits set to 1).

2.  **Q:** What is the purpose of the OOB area?
    *   **A:** Store metadata (ECC codes, bad block markers, filesystem info) separate from user data.

3.  **Q:** Why is ECC necessary for NAND?
    *   **A:** NAND cells degrade, causing bit flips. ECC detects and corrects these errors to ensure data integrity.

4.  **Q:** What happens when a block exceeds its erase cycle limit?
    *   **A:** It becomes unreliable, may fail to erase/program correctly, and should be marked as bad.

5.  **Q:** How does the BBT (Bad Block Table) work?
    *   **A:** Stores a list of bad block numbers, usually in the last good block. Checked before operations to skip bad blocks.

### Challenge Tasks

#### Challenge 1: "The ONFI Parser"
> **Task:** Implement ONFI parameter page parsing.
> *   Send 0xEC command.
> *   Read 256 bytes × 3 copies.
> *   Parse timing parameters.
> *   Auto-configure chip based on ONFI data.

#### Challenge 2: "The Wear Leveler"
> **Task:** Implement simple wear leveling.
> *   Track erase count per block.
> *   When writing, choose block with lowest count.
> *   Periodically move data from high-count to low-count blocks.

#### Challenge 3: "The Power-Loss Recovery"
> **Task:** Handle power loss during write.
> *   Implement atomic write (write to spare block, then update mapping).
> *   On mount, check for incomplete operations.
> *   Roll back or complete interrupted writes.

---

## 🔍 Deep Dive: NAND Read Operation

### Detailed Sequence

```c
// 1. Send READ command (0x00)
chip->legacy.cmd_ctrl(chip, NAND_CMD_READ0,
                     NAND_CTRL_CLE | NAND_CTRL_CHANGE);

// 2. Send column address (2 cycles for large page)
chip->legacy.cmd_ctrl(chip, column & 0xFF,
                     NAND_CTRL_ALE | NAND_CTRL_CHANGE);
chip->legacy.cmd_ctrl(chip, (column >> 8) & 0xFF,
                     NAND_CTRL_ALE);

// 3. Send row address (page number, 3 cycles)
chip->legacy.cmd_ctrl(chip, page & 0xFF,
                     NAND_CTRL_ALE);
chip->legacy.cmd_ctrl(chip, (page >> 8) & 0xFF,
                     NAND_CTRL_ALE);
chip->legacy.cmd_ctrl(chip, (page >> 16) & 0xFF,
                     NAND_CTRL_ALE);

// 4. Send READ START command (0x30)
chip->legacy.cmd_ctrl(chip, NAND_CMD_READSTART,
                     NAND_CTRL_CLE | NAND_CTRL_CHANGE);

// 5. Wait for R/B (Ready/Busy) signal
while (!chip->legacy.dev_ready(chip))
    cpu_relax();

// 6. Read data
chip->legacy.read_buf(chip, buf, mtd->writesize);

// 7. Read OOB
chip->legacy.read_buf(chip, oob_buf, mtd->oobsize);

// 8. Verify ECC
ret = chip->ecc.correct(chip, buf, oob_buf, ecc_calc);
if (ret < 0)
    pr_err("Uncorrectable ECC error\n");
```

---

## 📚 Further Reading & References

### Specifications
- ONFI (Open NAND Flash Interface) Specification
- JEDEC JESD230 (NAND Flash Specification)

### Kernel Documentation
- `Documentation/driver-api/mtdnand.rst`
- `Documentation/devicetree/bindings/mtd/nand-controller.yaml`

### Source Code
- `drivers/mtd/nand/raw/nand_base.c` - Core NAND functions
- `drivers/mtd/nand/raw/nand_bbt.c` - Bad block table
- `drivers/mtd/nand/raw/nand_ecc.c` - Software ECC
- `drivers/mtd/nand/raw/nand_bch.c` - BCH ECC

### Tools
- `mtd-utils` - NAND utilities
- `flashbench` - Flash benchmarking

---

## 🎓 Summary

Today we learned:

1.  **NAND Architecture:** Pages, blocks, planes, and LUNs.
2.  **NAND Commands:** READ, PROGRAM, ERASE, and their sequences.
3.  **Timing Requirements:** Critical parameters for reliable operation.
4.  **ECC Implementation:** Hamming, BCH, and on-die ECC.
5.  **Driver Structure:** Using MTD NAND framework.
6.  **Optimization:** DMA, multi-plane, cache programming.

**Key Takeaway:** NAND flash requires careful handling of timing, ECC, and bad blocks. The MTD NAND framework provides infrastructure, but driver must implement hardware-specific details correctly.

---

## 🚀 Next Steps

Tomorrow (Day 238), we'll complete Week 35 with a **Review and Project**, building a complete flash-based storage system with filesystem support.

---
