# Day 81: Audio Engine Implementation
## Phase 1: Core Embedded Engineering Foundations | Week 12: Capstone Project Phase 1

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
1.  **Implement** the `Audio_Task` as the highest priority thread in the system.
2.  **Design** a message-based architecture to control audio playback (Play, Stop, Volume).
3.  **Integrate** the DSP effects chain from Week 11 into the Capstone framework.
4.  **Develop** a "Clap Detector" algorithm using the microphone input.
5.  **Manage** audio buffers safely between ISR (DMA) and Task contexts.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   STM32F4 Discovery Board
    *   Microphone (On-board MP45DT02)
    *   Headphones
*   **Software Required:**
    *   VS Code with ARM GCC Toolchain
    *   PDM2PCM Library (Middleware from ST)
*   **Prior Knowledge:**
    *   Day 79 (BSP Audio)
    *   Day 75 (DMA Pipeline)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Audio Task Architecture
The Audio Task has two responsibilities:
1.  **Playback:** Refill the output buffer when DMA requests it.
2.  **Analysis:** Process microphone data to detect events (Claps).

**Constraint:** This task must run every ~10ms (Buffer Period). If it blocks for too long (e.g., waiting for a Mutex held by a low-priority task), audio glitches occur.
**Solution:**
*   Priority: **High (Real-Time)**.
*   Communication: **Queue**. Other tasks send commands (`CMD_PLAY_FILE`, `CMD_SET_VOL`). The Audio Task checks the queue *non-blocking* or with a very short timeout.

### 🔹 Part 2: PDM Microphone
The onboard mic is PDM (Pulse Density Modulation). It outputs a 1-bit stream at high frequency (e.g., 1-3 MHz).
*   **Decimation:** We need to convert this to PCM (16-bit, 48kHz).
*   **Library:** ST provides `libPDMFilter_CM4_GCC.a`.
*   **Flow:** I2S2 (Rx) -> DMA -> PDM Buffer -> PDM2PCM Filter -> PCM Buffer -> DSP Analysis.

---

## 💻 Implementation: app_audio.c

> **Instruction:** Implement the core audio logic.

### 👨‍💻 Code Implementation

#### Step 1: Definitions & Handles
```c
#include "app_audio.h"
#include "bsp_audio.h"
#include "FreeRTOS.h"
#include "queue.h"
#include "arm_math.h"

#define AUDIO_OUT_BUF_SIZE 2048
#define AUDIO_IN_BUF_SIZE  128 // PDM buffer

// Buffers (in CCM RAM if possible)
int16_t outBuffer[AUDIO_OUT_BUF_SIZE];
uint16_t pdmBuffer[AUDIO_IN_BUF_SIZE];
int16_t pcmBuffer[AUDIO_IN_BUF_SIZE/8]; // Decimation factor 64/8? Depends on config.

// Command Queue
typedef enum {
    AUDIO_CMD_NONE,
    AUDIO_CMD_PLAY_TONE,
    AUDIO_CMD_STOP,
    AUDIO_CMD_SET_VOL
} AudioCmdType_t;

typedef struct {
    AudioCmdType_t type;
    uint32_t param;
} AudioMsg_t;

QueueHandle_t hAudioQueue;
SemaphoreHandle_t hAudioBufferSem; // Signals DMA Half/Full
```

#### Step 2: Audio Task
```c
void vTaskAudio(void *p) {
    AudioMsg_t msg;
    hAudioQueue = xQueueCreate(10, sizeof(AudioMsg_t));
    hAudioBufferSem = xSemaphoreCreateBinary();
    
    // Init Hardware
    BSP_Audio_Init(48000);
    BSP_Mic_Init(48000); // Assume BSP_Mic exists
    
    // Start Playback (Silence initially)
    BSP_Audio_Play(outBuffer, AUDIO_OUT_BUF_SIZE);
    
    while(1) {
        // 1. Wait for DMA Interrupt Signal (Blocking)
        if (xSemaphoreTake(hAudioBufferSem, portMAX_DELAY)) {
            
            // 2. Check for Commands (Non-blocking)
            while (xQueueReceive(hAudioQueue, &msg, 0)) {
                Process_Command(&msg);
            }
            
            // 3. Refill Output Buffer
            // Determine which half needs refill based on state
            // (Simplified: Assume we track 'offset' globally or pass via Sem)
            Process_Playback();
            
            // 4. Process Input (Mic)
            // (Assume Mic DMA also signals or we poll)
            Process_Microphone();
        }
    }
}
```

#### Step 3: Command Processing
```c
static int isPlaying = 0;
static float toneFreq = 0;

void Process_Command(AudioMsg_t *msg) {
    switch(msg->type) {
        case AUDIO_CMD_PLAY_TONE:
            toneFreq = (float)msg->param;
            isPlaying = 1;
            break;
        case AUDIO_CMD_STOP:
            isPlaying = 0;
            memset(outBuffer, 0, sizeof(outBuffer)); // Clear
            break;
        case AUDIO_CMD_SET_VOL:
            BSP_Audio_SetVolume((uint8_t)msg->param);
            break;
    }
}
```

#### Step 4: Playback Logic (Synth)
```c
static float phase = 0;

void Process_Playback(void) {
    // Determine pointer: Lower or Upper half?
    // For this example, let's say we refill the *next* half.
    // Real implementation needs to know WHICH interrupt fired.
    
    int16_t *ptr = &outBuffer[current_offset]; 
    
    if (!isPlaying) {
        memset(ptr, 0, AUDIO_OUT_BUF_SIZE); // Silence
        return;
    }
    
    for(int i=0; i<AUDIO_OUT_BUF_SIZE/2; i+=2) {
        int16_t sample = (int16_t)(20000.0f * arm_sin_f32(phase));
        ptr[i] = sample;
        ptr[i+1] = sample;
        
        phase += 2*PI*toneFreq / 48000.0f;
        if (phase > 2*PI) phase -= 2*PI;
    }
}
```

---

## 💻 Implementation: Clap Detector

> **Instruction:** Detect a sudden spike in energy followed by decay.

### 👨‍💻 Code Implementation

#### Step 1: Energy Calculation
```c
#define CLAP_THRESHOLD 5000
#define CLAP_COOLDOWN  500 // ms

static uint32_t lastClapTime = 0;

void Process_Microphone(void) {
    // 1. Convert PDM to PCM (Library call)
    // PDM_Filter(pdmBuffer, pcmBuffer, &PDM_Filter_Handler);
    
    // 2. Calculate RMS
    float32_t rms;
    arm_rms_q15(pcmBuffer, AUDIO_IN_BUF_SIZE/8, &rms);
    
    // 3. Detect Spike
    if (rms > CLAP_THRESHOLD) {
        uint32_t now = xTaskGetTickCount();
        if ((now - lastClapTime) > CLAP_COOLDOWN) {
            lastClapTime = now;
            printf("CLAP DETECTED!\n");
            
            // Notify App Task
            AppMsg_t appMsg;
            appMsg.type = APP_CMD_CLAP;
            xQueueSend(hAppQueue, &appMsg, 0);
        }
    }
}
```

---

## 🔬 Lab Exercise: Lab 81.1 - The Clapper

### 1. Lab Objectives
- Integrate `vTaskAudio`.
- Clap hands to toggle LED.

### 2. Step-by-Step Guide

#### Phase A: Integration
1.  In `main.c`, ensure `vTaskAudio` is created.
2.  In `vTaskApp`, listen for `APP_CMD_CLAP`.
3.  If received, `BSP_LED_Toggle(LED_RED)`.

#### Phase B: Tuning
1.  Run with `printf` enabled for RMS values.
2.  Observe noise floor (e.g., 500).
3.  Clap. Observe peak (e.g., 8000).
4.  Set `CLAP_THRESHOLD` to 4000.

### 3. Verification
If false triggers occur, increase threshold. If misses occur, decrease. Ensure Mic Gain is set correctly in `BSP_Mic_Init`.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Double Clap
- **Goal:** Smarter detection.
- **Task:**
    1.  State Machine: `WAIT_CLAP_1` -> `WAIT_CLAP_2` (within 1s).
    2.  If 2 claps detected, toggle Light.
    3.  If only 1, timeout and reset.

### Lab 3: Audio Notification
- **Goal:** Feedback.
- **Task:**
    1.  When Clap detected, send `AUDIO_CMD_PLAY_TONE` (1kHz, 200ms) to Audio Queue.
    2.  Hear a "Beep" when you clap.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stack Overflow in Audio Task
*   **Cause:** `arm_sin_f32` or PDM library using stack?
*   **Solution:** Increase stack to 1024 words.

#### 2. HardFault (Usage Fault)
*   **Cause:** FPU disabled in ISR context?
*   **Solution:** Ensure FPU is enabled globally.

#### 3. Glitching Audio
*   **Cause:** `printf` in `Process_Microphone`.
*   **Solution:** `printf` is slow/blocking! Remove it or use a very fast non-blocking logger.

---

## ⚡ Optimization & Best Practices

### Code Quality
- **Event Bits:** Instead of a Semaphore for DMA, use `xTaskNotifyFromISR` with bits indicating `TX_HALF`, `TX_CPLT`, `RX_HALF`, `RX_CPLT`. This allows handling multiple events in one loop.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we process audio in the Task and not the ISR?
    *   **A:** ISRs should be short. DSP processing (RMS, Filtering) takes time. Doing it in a High Priority Task allows interrupts (like SysTick or Ethernet) to still fire, while ensuring Audio preempts lower priority tasks.
2.  **Q:** How does PDM differ from PCM?
    *   **A:** PDM is density of pulses (1-bit). PCM is amplitude value (16-bit). PDM requires digital filtering to recover the analog signal.

### Challenge Task
> **Task:** Implement "Voice Activity Detection" (VAD). Instead of simple RMS, use Zero Crossing Rate + Energy to distinguish Speech from Noise.

---

## 📚 Further Reading & References
- [ST AN3998: PDM audio software decoding](https://www.st.com/resource/en/application_note/dm00040920-pdm-audio-software-decoding-on-stm32-microcontrollers-stmicroelectronics.pdf)

---
