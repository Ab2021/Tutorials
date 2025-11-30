# Day 169: Week 24 Review & Project - The Virtual Sound Card
## Phase 2: Linux Kernel & Device Drivers | Week 24: ALSA Audio Subsystem

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
1.  **Synthesize** Week 24 concepts (ALSA, PCM, Controls, ASoC).
2.  **Architect** a complete Audio Driver stack.
3.  **Implement** a Virtual Sound Card that generates tones (Sine/Square) on playback.
4.  **Implement** Mixer controls to change the tone frequency and volume.
5.  **Verify** the driver using `aplay`, `arecord`, and `audacity`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `audacity` (optional, for visualization).
*   **Prior Knowledge:**
    *   Week 24 Content.

---

## 🔄 Week 24 Review

### 1. ALSA Core (Day 163)
*   **Card:** The container.
*   **Device:** The logical unit.

### 2. PCM Interface (Day 164)
*   **Ops:** `open`, `trigger`, `pointer`.
*   **Buffer:** Ring buffer management.

### 3. Controls (Day 165)
*   **Mixer:** Volume, Mute, Enum.
*   **Callbacks:** `info`, `get`, `put`.

### 4. ASoC (Day 166-167)
*   **Split:** Machine, Platform, Codec.
*   **DAPM:** Power management graph.

### 5. Jack Detection (Day 168)
*   **Event:** `EV_SW` via Input Subsystem.

---

## 🛠️ Project: The "SynthCard"

### 📋 Project Requirements
Create a driver `synthcard` that:
1.  **Registers** a Sound Card.
2.  **Playback:** Accepts audio data (and ignores it, or visualizes it).
3.  **Capture:** Generates a synthetic waveform (Sine Wave).
4.  **Controls:**
    *   "Frequency" (Slider 100Hz - 1000Hz).
    *   "Waveform" (Enum: Sine, Square, Triangle).
5.  **Jack:** Simulates a Headphone insertion via a sysfs trigger.

---

## 💻 Implementation: Step-by-Step Guide

### 🔹 Phase 1: Header & Structs

**`synthcard.h`**
```c
#ifndef SYNTHCARD_H
#define SYNTHCARD_H

#include <sound/core.h>
#include <sound/pcm.h>
#include <sound/control.h>
#include <linux/module.h>

struct synth_dev {
    struct snd_card *card;
    struct snd_pcm *pcm;
    
    // Generator State
    int freq;
    int waveform; // 0=Sine, 1=Square
    u32 phase;
    
    // PCM State
    struct snd_pcm_substream *capture_substream;
    struct hrtimer timer;
    int running;
    u32 pos; // Buffer position in frames
    
    spinlock_t lock;
};

#endif
```

### 🔹 Phase 2: The Generator (Capture)

**`synth_gen.c`**
```c
#include "synthcard.h"
#include <linux/math64.h>

// Simple Sine Approximation or Lookup Table
static s16 generate_sample(struct synth_dev *chip) {
    // Phase accumulator logic
    // ...
    return (s16)value;
}

static void fill_capture_buffer(struct synth_dev *chip, int frames) {
    struct snd_pcm_runtime *runtime = chip->capture_substream->runtime;
    void *dma_area = runtime->dma_area + frames_to_bytes(runtime, chip->pos);
    s16 *samples = (s16 *)dma_area;
    int i;
    
    for (i = 0; i < frames; i++) {
        s16 val = generate_sample(chip);
        samples[2*i] = val;     // Left
        samples[2*i+1] = val;   // Right
    }
}
```

### 🔹 Phase 3: PCM Ops

**`synth_pcm.c`**
```c
// Timer Callback
static enum hrtimer_restart synth_timer_callback(struct hrtimer *timer) {
    struct synth_dev *chip = container_of(timer, struct synth_dev, timer);
    unsigned long flags;
    
    spin_lock_irqsave(&chip->lock, flags);
    if (!chip->running) {
        spin_unlock_irqrestore(&chip->lock, flags);
        return HRTIMER_NORESTART;
    }
    
    // Fill 1 period
    int frames = chip->capture_substream->runtime->period_size;
    fill_capture_buffer(chip, frames);
    
    // Update Position
    chip->pos += frames;
    if (chip->pos >= chip->capture_substream->runtime->buffer_size)
        chip->pos = 0;
        
    snd_pcm_period_elapsed(chip->capture_substream);
    
    spin_unlock_irqrestore(&chip->lock, flags);
    hrtimer_forward_now(timer, ktime_set(0, period_ns));
    return HRTIMER_RESTART;
}
```

### 🔹 Phase 4: Controls

**`synth_ctl.c`**
```c
static int synth_freq_info(struct snd_kcontrol *kcontrol, struct snd_ctl_elem_info *uinfo) {
    uinfo->type = SNDRV_CTL_ELEM_TYPE_INTEGER;
    uinfo->count = 1;
    uinfo->value.integer.min = 100;
    uinfo->value.integer.max = 2000;
    return 0;
}
// ... get/put ...
```

---

## 💻 Implementation: Testing

> **Instruction:** Compile and Load.

### 👨‍💻 Command Line Steps

1.  **Load:** `insmod synthcard.ko`
2.  **Record:**
    ```bash
    arecord -D hw:1,0 -f S16_LE -r 48000 -c 2 -d 5 test.wav
    ```
3.  **Change Freq (while recording):**
    ```bash
    amixer -c 1 cset name='Frequency' 500
    ```
4.  **Play Back:**
    ```bash
    aplay test.wav
    ```
    *   You should hear a tone that changes pitch.

---

## 📈 Grading Rubric

| Criteria | Excellent (A) | Good (B) | Needs Improvement (C) |
| :--- | :--- | :--- | :--- |
| **Stability** | No XRUNs under load. Clean unload. | Occasional glitches. | Kernel Panic. |
| **Functionality** | Frequency control works in real-time. | Frequency fixed. | No sound. |
| **Code Quality** | Modular, proper locking. | Monolithic. | Race conditions. |

---

## 🔮 Looking Ahead: Phase 3
Next week, we start **Phase 3: Embedded Android**.
*   We will leave the Kernel (mostly) and move to Userspace.
*   We will learn how Android uses these drivers (HALs).

---
