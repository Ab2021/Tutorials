# Day 165: ALSA Control Interface (Mixer)
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
1.  **Define** ALSA Controls (`snd_kcontrol_new`).
2.  **Implement** the 3 callbacks: `info`, `get`, `put`.
3.  **Create** standard controls: "Master Playback Volume" and "Master Playback Switch" (Mute).
4.  **Register** controls to the card (`snd_ctl_add`).
5.  **Manipulate** controls using `amixer` and `alsamixer`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `alsamixer`.
*   **Prior Knowledge:**
    *   Day 163 (ALSA Card).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Mixer Abstraction
Users want to change volume, mute channels, or select inputs.
ALSA provides a generic "Control" interface.
*   **Element:** A single control (e.g., "Master Volume").
*   **Type:** Boolean (Switch), Integer (Volume), Enumerated (Input Select), Bytes (DSP Data).
*   **Access:** Read, Write, Volatile.

### 🔹 Part 2: The Callbacks
1.  **Info:** Describes the control (Min, Max, Step, Name).
2.  **Get:** Returns the current value (Driver -> User).
3.  **Put:** Updates the value (User -> Driver). Returns 1 if changed, 0 if same.

---

## 💻 Implementation: Volume Control

> **Instruction:** Add a Master Volume control (0-100).

### 👨‍💻 Code Implementation

#### Step 1: Data Storage
```c
struct my_audio_dev {
    // ...
    int volume;
    spinlock_t mixer_lock;
};
```

#### Step 2: Callbacks
```c
// INFO: Describe the control
static int my_vol_info(struct snd_kcontrol *kcontrol, struct snd_ctl_elem_info *uinfo) {
    uinfo->type = SNDRV_CTL_ELEM_TYPE_INTEGER;
    uinfo->count = 1; // Mono (use 2 for Stereo)
    uinfo->value.integer.min = 0;
    uinfo->value.integer.max = 100;
    return 0;
}

// GET: Read current value
static int my_vol_get(struct snd_kcontrol *kcontrol, struct snd_ctl_elem_value *ucontrol) {
    struct my_audio_dev *chip = snd_kcontrol_chip(kcontrol);
    unsigned long flags;

    spin_lock_irqsave(&chip->mixer_lock, flags);
    ucontrol->value.integer.value[0] = chip->volume;
    spin_unlock_irqrestore(&chip->mixer_lock, flags);
    return 0;
}

// PUT: Write new value
static int my_vol_put(struct snd_kcontrol *kcontrol, struct snd_ctl_elem_value *ucontrol) {
    struct my_audio_dev *chip = snd_kcontrol_chip(kcontrol);
    unsigned long flags;
    int changed = 0;
    int val = ucontrol->value.integer.value[0];

    if (val < 0 || val > 100) return -EINVAL;

    spin_lock_irqsave(&chip->mixer_lock, flags);
    if (chip->volume != val) {
        chip->volume = val;
        // update_hardware_volume(chip, val);
        changed = 1;
    }
    spin_unlock_irqrestore(&chip->mixer_lock, flags);
    
    return changed;
}
```

#### Step 3: Definition & Registration
```c
static const struct snd_kcontrol_new my_vol_control = {
    .iface = SNDRV_CTL_ELEM_IFACE_MIXER,
    .name = "Master Playback Volume",
    .info = my_vol_info,
    .get = my_vol_get,
    .put = my_vol_put,
};

// In Probe:
spin_lock_init(&chip->mixer_lock);
ret = snd_ctl_add(card, snd_ctl_new1(&my_vol_control, chip));
```

---

## 💻 Implementation: Mute Switch (Boolean)

> **Instruction:** Add a Mute Switch.

### 👨‍💻 Code Implementation

```c
static int my_switch_info(struct snd_kcontrol *kcontrol, struct snd_ctl_elem_info *uinfo) {
    uinfo->type = SNDRV_CTL_ELEM_TYPE_BOOLEAN;
    uinfo->count = 1;
    uinfo->value.integer.min = 0;
    uinfo->value.integer.max = 1;
    return 0;
}

// Get/Put similar to Volume, but using chip->mute
// Name: "Master Playback Switch"
```

---

## 🔬 Lab Exercise: Lab 165.1 - Testing Mixer

### 1. Lab Objectives
- Compile and load.
- Use `amixer` to read/write.
- Use `alsamixer` GUI.

### 2. Step-by-Step Guide
1.  **Check Controls:**
    ```bash
    amixer -c 1 contents
    ```
2.  **Set Volume:**
    ```bash
    amixer -c 1 cset numid=1 50
    ```
3.  **GUI:**
    ```bash
    alsamixer -c 1
    ```
    *   You should see a bar for Volume and a box for Mute.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Stereo Volume
- **Goal:** Left/Right control.
- **Task:**
    1.  Change `uinfo->count = 2`.
    2.  Update `get`/`put` to handle `value[0]` (Left) and `value[1]` (Right).
    3.  Store `int volume[2]` in chip struct.

### Lab 3: Enumerated Control (Mux)
- **Goal:** Select Input (Mic vs Line).
- **Task:**
    1.  `uinfo->type = SNDRV_CTL_ELEM_TYPE_ENUMERATED`.
    2.  `snd_ctl_enum_info(uinfo, 1, 2, texts);` where `texts` is `{"Mic", "Line"}`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Control not found"
*   **Cause:** `snd_ctl_add` failed.
*   **Cause:** Name mismatch. ALSA has strict naming conventions ("Master", "PCM", "Headphone").

#### 2. alsamixer doesn't show dB
*   **Cause:** You need to provide TLV (Type-Length-Value) data to map 0-100 to Decibels.
*   **Fix:** Use `DECLARE_TLV_DB_SCALE` and `.tlv.p` in `snd_kcontrol_new`.

---

## ⚡ Optimization & Best Practices

### Standard Names
*   Use standard names defined in `sound/core.h` or conventions.
*   "Master Playback Volume" -> Controls global volume.
*   "PCM Playback Volume" -> Controls digital stream volume.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does `put` return 1 or 0?
    *   **A:** If it returns 1, the ALSA core sends a notification event to userspace (e.g., alsamixer updates its GUI). If 0, no event is sent.
2.  **Q:** What is `SNDRV_CTL_ELEM_IFACE_MIXER`?
    *   **A:** It defines the "Interface". Mixer is for user controls. `CARD` is for global card settings. `PCM` is for stream-specific settings.

### Challenge Task
> **Task:** "The Bass Boost".
> *   Add a boolean control "Bass Boost Switch".
> *   When enabled, print a kernel message "Bass Boost ON".

---

## 📚 Further Reading & References
- [Kernel Documentation: sound/kernel-api/writing-an-alsa-driver.html#control-interface](https://www.kernel.org/doc/html/latest/sound/kernel-api/writing-an-alsa-driver.html#control-interface)

---
