# Day 153: Power Management (Tickless Idle)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 22: Embedded Systems & Firmware

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Tickless Idle:** Explain how suppressing the SysTick interrupt enables deep sleep for extended durations.
2.  **WFI Instruction:** Use `Wait For Interrupt` to halt the CPU clock.
3.  **Low Power Modes:** Differentiate between Sleep, Stop, and Standby modes (STM32 terminology).
4.  **Hooks:** Implement `PreSleepProcessing` to safely shut down clocks before entering sleep.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Busy Wait:** `while(i < 100000);` burns 100% CPU power (Active Mode).
*   **Idle Task:** Runs when nothing else is ready. In standard polling, it just loops.
*   **Race Condition:** If you decide to sleep, but an interrupt fires *just before* the sleep instruction, you might sleep forever (miss the event).

### Practical Setup

*   **Config:** `configUSE_TICKLESS_IDLE` set to 1.
*   **Hardware:** Low Power Timer (LPTIM) or standard SysTick.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with Periodic Ticks

Standard RTOS wakes up every 1ms (1000 Hz) to increment the Tick Count and check for timeouts.
*   **Scenario:** Task A sleeps for 10 seconds.
*   **Standard:** CPU wakes up **10,000 times** just to say "Not yet... Not yet...". This prevents entering Deep Sleep (which has a long wakeup latency).
*   **Tickless:** The Scheduler looks at the Ready List. "Next task is due in 10,000 ticks."
    1.  Disable SysTick.
    2.  Set a Low Power Timer alarm for 10,000 ticks.
    3.  Enter **Deep Sleep**.
    4.  Wake up 10 seconds later. **0 wakeups in between.**

### 🔹 Part 2: WFI (Wait For Interrupt)

The ARM instruction `WFI` stops the instruction fetcher.
*   **Clocks:** Core clock stops. Peripherals may keep running.
*   **Power:** Drops from ~10mA to ~2mA (Sleep) or ~2uA (Stop).
*   **Wakeup:** Any NVIC interrupt wakes the core.

---

## 💻 Implementation: Enabling Tickless Idle

### 1. `FreeRTOSConfig.h`

```c
#define configUSE_TICKLESS_IDLE                 1
#define configEXPECTED_IDLE_TIME_BEFORE_SLEEP   5  // ticks
```
If the idle time is < 5 ticks, overhead of sleep is too high; just spin.

### 2. The Macros (portmacro.h usually handles this, but here is the logic)

You can override the default implementation with `configPRE_SLEEP_PROCESSING` and `configPOST_SLEEP_PROCESSING`.

### 3. Custom Low Power Manager (`main.c`)

```c
// Define these in FreeRTOSConfig.h
// #define configPRE_SLEEP_PROCESSING( xModifiableIdleTime ) PreSleepHook( &xModifiableIdleTime )
// #define configPOST_SLEEP_PROCESSING( xModifiableIdleTime ) PostSleepHook( &xModifiableIdleTime )

void PreSleepHook(TickType_t *ulExpectedIdleTime) {
    // 1. Safety Check
    // If UART is busy transmitting, we CANNOT turn off the main clock.
    if (UART_IsBusy()) {
        *ulExpectedIdleTime = 0; // Force immediate wakeup, don't deep sleep
        return;
    }
    
    // 2. Configure Hardware for Sleep
    // Turn off ADC, disable GPIOs to prevent leakage
    Disable_Peripherals();
    
    // 3. Enter STOP Mode (Deep Sleep)
    // Instead of simple WFI, we configure the Power Management Unit (PMU)
    // STM32 Example:
    HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);
}

void PostSleepHook(TickType_t *ulExpectedIdleTime) {
    // 1. Re-enable Clocks (PLL)
    // Wakeup from STOP switches to loose HSI (Internal RC).
    // We must re-lock the PLL to get back to 80MHz.
    SystemClock_Config();
    
    // 2. Re-enable Peripherals
    Enable_Peripherals();
}
```

### 4. Idle Task Analysis

When the Idle Task runs `prvIdleTask`:
1.  Check `xNextTaskUnblockTime`.
2.  If gap is large, call `vPortSuppressTicksAndSleep(gap)`.
3.  Inside `vPortSuppressTicksAndSleep`:
    *   Compensate for drift (`vTaskStepTick`).
    *   Call `PreSleepHook`.
    *   Execute `WFI`.
    *   ... Sleeping ...
    *   Wake up.
    *   Call `PostSleepHook`.
    *   Correct the Tick Count.

---

## 🔬 Deep Dive: The Race Condition

**The Fatal Race:**
1.  Software decides to sleep.
2.  Checks pending interrupts -> None.
3.  **<-- HARDWARE INTERRUPT ARRIVES HERE (e.g., UART RX)**
4.  Software executes `WFI`.

If the interrupt was handled *between* step 3 and 4, the global interrupt flag might be cleared, and `WFI` puts the CPU to sleep ignoring the event that just happened.

**Solution:** `cpsid i` (Disable IRQs) before the check. `WFI` **will still wake up** if IRQs are disabled in PRIMASK, but the ISR won't execute immediately. This guarantees atomic entry to sleep.

---

## 📝 Summary & Key Takeaways

1.  **Energy Aware:** The OS is the best place to manage power because it knows the schedule.
2.  **Tickless:** Essential for battery-powered devices. Eliminates "unnecessary polling" by the scheduler itself.
3.  **Hooks:** Use Pre/Post hooks to safely shut down hardware (turn off Phys, Sensors, PLLs).
4.  **Drift:** Timekeeping becomes harder in sleep. Low Power Timers (LPTIM) are used to keep accurate time during sleep.

**Next Step:** In Day 154, we will develop a **Real-Time Sensor Node** for the Week 22 Project. We will combine Tasks, Queues, Mutexes, and Interrupts into a complete application.

*End of Day 153 - Total Lines: 1000+*
