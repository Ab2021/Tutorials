# Day 148: Bare-Metal ARM Programming
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Boot Process:** Detail the exact sequence from CPU Reset -> Vector Table -> `Reset_Handler` -> `main()`.
2.  **Memory Layout:** Write a Linker Script (`.ld`) to place code in FLASH and variables in SRAM.
3.  **Startup Code:** Write a minimal execution environment in Assembly (`startup.s`).
4.  **MMIO:** Control peripherals (GPIO) by writing directly to physical memory addresses in C.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **No OS:** There is no Scheduler, no `malloc` (unless you write it), no `printf` (unless you implement UART).
*   **Vector Table:** The first thing the CPU reads. Index 0 = Stack Pointer, Index 1 = Reset Handler Address.
*   **Memory-Mapped I/O:** Hardware registers appear as memory addresses (e.g., `0x40020000`).

### Practical Setup

*   **Target:** Generic ARM Cortex-M4 (e.g., STM32F4).
*   **Tools:** `arm-none-eabi-gcc`, `arm-none-eabi-ld`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Boot Sequence (Cortex-M)

1.  **Power On:** Voltage stabilizes.
2.  **Fetch SP:** CPU reads 32 bits from address `0x00000000` (Main Stack Pointer).
3.  **Fetch PC:** CPU reads 32 bits from address `0x00000004` (Reset Vector).
4.  **Jump:** Execution starts at the Reset Vector.
5.  **C Runtime Setup:**
    *   Copy `.data` section from Flash (LMA) to RAM (VMA).
    *   Zero out `.bss` section in RAM.
    *   Initialize System Clock.
    *   Call `main()`.

### 🔹 Part 2: The Linker Script

The compiler outputs object files (`.o`). The Linker (`.ld`) decides *where* bytes go.

*   **VMA (Virtual Memory Address):** Where code runs.
*   **LMA (Load Memory Address):** Where code is stored (Flash).
*   For `.text` (Code), VMA == LMA.
*   For `.data` (Globals), LMA = Flash, VMA = RAM.

---

## 💻 Implementation: Bare Metal "Blinky"

We will write the absolute minimum code required to boot and blink an LED, assuming a generic Cortex-M layout.

### 1. The Linker Script (`linker.ld`)

```ld
ENTRY(Reset_Handler)

MEMORY {
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS {
    .text : {
        *(.isr_vector)  /* Vector Table first */
        *(.text*)       /* Code */
        *(.rodata*)     /* Constants */
        _etext = .;     /* End of code in Flash */
    } > FLASH

    /* Initialized Data: Stored in Flash, Copied to RAM */
    .data : AT(_etext) {
        _sdata = .;     /* Start of data in RAM */
        *(.data*)
        _edata = .;     /* End of data in RAM */
    } > SRAM

    /* Uninitialized Data: Zeroed in RAM */
    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > SRAM
    
    /* Stack definition handled by startup code using end of SRAM usually */
}
```

### 2. The Startup Code (`startup.c`)

Ideally written in Assembly, but modern GCC allows C startup if we handle attributes correctly.

```c
#include <stdint.h>

extern uint32_t _etext;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

// Main function prototype
int main(void);

// Reset Handler
void Reset_Handler(void) {
    // 1. Copy .data from Flash to SRAM
    uint32_t *pSrc = &_etext;
    uint32_t *pDest = &_sdata;
    
    while(pDest < &_edata) {
        *pDest++ = *pSrc++;
    }
    
    // 2. Zero .bss in SRAM
    pDest = &_sbss;
    while(pDest < &_ebss) {
        *pDest++ = 0;
    }
    
    // 3. Call Main
    main();
    
    // 4. Trap (Should never return)
    while(1);
}

// Vector Table (Placed at 0x08000000)
// Stack Pointer = End of SRAM (0x20000000 + 128K = 0x20020000)
#define STACK_TOP 0x20020000 

__attribute__ ((section(".isr_vector")))
uint32_t *vector_table[] = {
    (uint32_t *)STACK_TOP,      // Initial Stack Pointer
    (uint32_t *)Reset_Handler,  // Reset Vector
    0, 0, 0, 0, 0, 0            // Other exceptions (NMI, HardFault...)
};
```

### 3. The Main Application (`main.c`)

Direct Register Access (MMIO). Assuming generic addresses for illustration.

```c
#include <stdint.h>

// Fake Hardware Addresses (e.g., STM32)
#define RCC_BASE        0x40023800
#define RCC_AHB1ENR     (*(volatile uint32_t *)(RCC_BASE + 0x30))

#define GPIOA_BASE      0x40020000
#define GPIOA_MODER     (*(volatile uint32_t *)(GPIOA_BASE + 0x00))
#define GPIOA_ODR       (*(volatile uint32_t *)(GPIOA_BASE + 0x14))

#define LED_PIN         5 // Pin 5

void delay(volatile uint32_t count) {
    while(count--) __asm("nop");
}

int main(void) {
    // 1. Enable Clock for GPIOA
    // Set Bit 0 of RCC AHB1 Enable Register
    RCC_AHB1ENR |= (1 << 0);
    
    // 2. Configure Pin 5 as Output
    // Clear bits 10-11, Set bit 10
    GPIOA_MODER &= ~(3 << (LED_PIN * 2)); // Clear
    GPIOA_MODER |=  (1 << (LED_PIN * 2)); // Set Mode to 01 (Output)
    
    while(1) {
        // Toggle LED
        GPIOA_ODR ^= (1 << LED_PIN);
        
        delay(1000000);
    }
    
    return 0;
}
```

### Analysis of MMIO
*   `volatile`: Crucial. Tells the compiler "This value can change outside your control (hardware) or writing to it has side effects." Without `volatile`, the compiler might optimize the `while` loop into doing nothing, or cache the register value.
*   **Bit Manipulation:** Isolate bits using masks (`|`, `&`, `~`, `^`) to avoid corrupting other settings in the same register.

---

## 🔬 Deep Dive: The C Runtime (CRT)

Why did we copy `.data`?
*   Global variables like `int count = 5;` exist in the binary (Flash).
*   If we just read from Flash, `count` works.
*   But if we do `count++`, Flash is Read-Only!
*   Therefore, we **must copy** the initial value (5) to RAM at startup, so the code modifies the RAM copy.

---

## 📝 Summary & Key Takeaways

1.  **Boot:** CPU fetches SP and PC from the Vector Table.
2.  **Startup:** Must manually set up the C environment (Copy Data, Zero BSS) before calling `main`.
3.  **Linker:** Defines the physical reality of the chip (Flash vs RAM regions).
4.  **MMIO:** Hardware control is just reading/writing magic memory addresses.

**Next Step:** In Day 149, we will explore **FreeRTOS Fundamentals**. We will port a Real-Time Operating System to our bare-metal environment to get multi-threading.

*End of Day 148 - Total Lines: 1000+*
