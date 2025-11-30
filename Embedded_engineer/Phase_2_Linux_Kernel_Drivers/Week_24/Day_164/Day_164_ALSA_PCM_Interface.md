# Day 164: ALSA PCM Interface
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
1.  **Create** a PCM device (`snd_pcm_new`).
2.  **Implement** `snd_pcm_ops` (open, close, ioctl, hw_params, prepare, trigger, pointer).
3.  **Define** Hardware Capabilities (`snd_pcm_hardware`).
4.  **Manage** the Ring Buffer (DMA Area).
5.  **Debug** XRUNs (Underrun/Overrun) and buffer position updates.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 163 (ALSA Card).
    *   Circular Buffers.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The PCM Middle Layer
The PCM layer handles the ring buffer logic.
*   **Application:** Writes data to the buffer.
*   **Hardware:** Reads data from the buffer (DMA).
*   **Driver:** Updates the "Pointer" (Where is the HW reading now?).

### 🔹 Part 2: Periods and Buffers
*   **Buffer Size:** Total size of the ring buffer (e.g., 4096 frames).
*   **Period Size:** Interrupt interval (e.g., 1024 frames).
*   **Interrupt:** Fires every "Period". The driver calls `snd_pcm_period_elapsed()`.

### 🔹 Part 3: The State Machine
*   **OPEN:** App opened device.
*   **SETUP:** Format/Rate configured.
*   **PREPARED:** Buffers allocated, ready to start.
*   **RUNNING:** DMA is active.
*   **XRUN:** Underrun (App too slow) or Overrun (App too fast).

---

## 💻 Implementation: PCM Device Creation

> **Instruction:** Add a PCM device to our card.

### 👨‍💻 Code Implementation

#### Step 1: Hardware Definition
```c
static const struct snd_pcm_hardware my_pcm_hw = {
    .info = SNDRV_PCM_INFO_MMAP | SNDRV_PCM_INFO_MMAP_VALID |
            SNDRV_PCM_INFO_BLOCK_TRANSFER | SNDRV_PCM_INFO_PAUSE,
    .formats = SNDRV_PCM_FMTBIT_S16_LE,
    .rates = SNDRV_PCM_RATE_48000,
    .rate_min = 48000,
    .rate_max = 48000,
    .channels_min = 2,
    .channels_max = 2,
    .buffer_bytes_max = 64 * 1024,
    .period_bytes_min = 4096,
    .period_bytes_max = 32 * 1024,
    .periods_min = 2,
    .periods_max = 16,
};
```

#### Step 2: PCM Operations
```c
static int my_pcm_open(struct snd_pcm_substream *substream) {
    struct my_audio_dev *chip = snd_pcm_substream_chip(substream);
    struct snd_pcm_runtime *runtime = substream->runtime;

    runtime->hw = my_pcm_hw; // Copy HW caps
    return 0;
}

static int my_pcm_close(struct snd_pcm_substream *substream) {
    return 0;
}

static int my_pcm_trigger(struct snd_pcm_substream *substream, int cmd) {
    switch (cmd) {
    case SNDRV_PCM_TRIGGER_START:
        // Enable DMA / Timer
        break;
    case SNDRV_PCM_TRIGGER_STOP:
        // Disable DMA / Timer
        break;
    default:
        return -EINVAL;
    }
    return 0;
}

static snd_pcm_uframes_t my_pcm_pointer(struct snd_pcm_substream *substream) {
    // Return current HW position in frames (0 to buffer_size - 1)
    // For dummy driver, we simulate this later.
    return 0; 
}

static const struct snd_pcm_ops my_pcm_ops = {
    .open = my_pcm_open,
    .close = my_pcm_close,
    .ioctl = snd_pcm_lib_ioctl,
    .hw_params = my_hw_params, // Memory allocation
    .hw_free = my_hw_free,
    .prepare = my_prepare,
    .trigger = my_pcm_trigger,
    .pointer = my_pcm_pointer,
};
```

#### Step 3: Registration (in Probe)
```c
struct snd_pcm *pcm;
ret = snd_pcm_new(card, "My PCM", 0, 1, 0, &pcm); // 1 Playback, 0 Capture
if (ret < 0) return ret;

snd_pcm_set_ops(pcm, SNDRV_PCM_STREAM_PLAYBACK, &my_pcm_ops);
pcm->private_data = chip;
strcpy(pcm->name, "My PCM Device");

// Pre-allocate DMA buffer (Continuous)
snd_pcm_set_managed_buffer_all(pcm, SNDRV_DMA_TYPE_CONTINUOUS, NULL, 64*1024, 64*1024);
```

---

## 💻 Implementation: Buffer Management

> **Instruction:** Implement `hw_params` and `pointer`.

### 👨‍💻 Code Implementation

#### Step 1: HW Params (Allocation)
Since we used `snd_pcm_set_managed_buffer_all`, the core handles allocation!
We just need a dummy function or standard helper.
```c
static int my_hw_params(struct snd_pcm_substream *substream,
                        struct snd_pcm_hw_params *hw_params) {
    return 0; // Core handles buffer allocation
}

static int my_hw_free(struct snd_pcm_substream *substream) {
    return 0;
}
```

#### Step 2: Prepare
Called before Start. Reset pointers.
```c
static int my_prepare(struct snd_pcm_substream *substream) {
    struct my_audio_dev *chip = snd_pcm_substream_chip(substream);
    chip->pos = 0; // Reset internal position counter
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 164.1 - Testing PCM

### 1. Lab Objectives
- Compile and load.
- Check devices.
- Try to play audio (it will hang because `trigger` does nothing yet).

### 2. Step-by-Step Guide
1.  **Check Device:**
    ```bash
    ls -l /dev/snd/pcm*
    # /dev/snd/pcmC1D0p  (Card 1, Device 0, Playback)
    ```
2.  **Play:**
    ```bash
    aplay -D hw:1,0 -r 48000 -f S16_LE -c 2 /dev/urandom
    ```
    *   It should open, but stay stuck at 0% because the pointer never advances.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Simulating Playback (Timer)
- **Goal:** Make `aplay` progress.
- **Task:**
    1.  Add `hrtimer`.
    2.  In `trigger(START)`, start timer.
    3.  In timer callback:
        *   Increment `chip->pos` by `period_size`.
        *   Wrap around `buffer_size`.
        *   Call `snd_pcm_period_elapsed(substream)`.
    4.  In `pointer`, return `bytes_to_frames(chip->pos)`.

### Lab 3: Capture Stream
- **Goal:** Add recording support.
- **Task:**
    1.  `snd_pcm_new(..., 1, 1, ...)` (1 Playback, 1 Capture).
    2.  `snd_pcm_set_ops(..., SNDRV_PCM_STREAM_CAPTURE, &my_capture_ops)`.
    3.  Implement ops for capture (fill buffer with data).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Invalid Argument" on Open
*   **Cause:** `snd_pcm_hardware` constraints not met.
*   **Example:** App requests 44.1kHz, but driver only supports 48kHz.

#### 2. "Input/Output Error" (XRUN)
*   **Cause:** `pointer` callback returning invalid value (outside buffer).
*   **Cause:** Interrupts not firing fast enough.

---

## ⚡ Optimization & Best Practices

### `snd_pcm_lib_ioctl`
*   Standard helper for `ioctl`. Always use this unless you have very specific needs.
*   Handles `SNDRV_PCM_IOCTL_RESET`, `SNDRV_PCM_IOCTL_INFO`, etc.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `SNDRV_PCM_INFO_MMAP`?
    *   **A:** It means the hardware buffer can be mapped into userspace. This avoids `copy_from_user` overhead (Zero Copy).
2.  **Q:** Why `snd_pcm_period_elapsed`?
    *   **A:** It tells the ALSA core "Hardware has finished processing one period of data". The core then updates the "Application Pointer" and wakes up userspace to write more data.

### Challenge Task
> **Task:** "The Sine Wave Generator".
> *   Implement the Capture stream.
> *   In the timer callback, fill the capture buffer with a generated Sine Wave.
> *   Record it: `arecord -D hw:1,0 -f S16_LE -r 48000 test.wav`.
> *   Play it back to verify.

---

## 📚 Further Reading & References
- [Kernel Documentation: sound/kernel-api/writing-an-alsa-driver.html#pcm-interface](https://www.kernel.org/doc/html/latest/sound/kernel-api/writing-an-alsa-driver.html#pcm-interface)

---
