# Day 168: ALSA Jack Detection & UCM
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
1.  **Implement** Jack Detection (`snd_soc_jack`) in the Machine Driver.
2.  **Report** insertion/removal events (`snd_soc_jack_report`).
3.  **Integrate** with GPIOs (`snd_soc_jack_add_gpios`).
4.  **Understand** UCM (Use Case Manager) configuration files.
5.  **Debug** jack events using `evtest` (Input Subsystem).

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   `evtest`.
*   **Prior Knowledge:**
    *   Day 166 (ASoC).
    *   Day 147 (Input Subsystem).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Jack Detection
When you plug in headphones:
1.  **Hardware:** Detect pin goes High/Low.
2.  **Driver:** Interrupt fires.
3.  **ASoC:** `snd_soc_jack_report` updates the status.
4.  **Input:** An `EV_SW` event (`SW_HEADPHONE_INSERT`) is sent to userspace.
5.  **PulseAudio/PipeWire:** Switches output from Speaker to Headphone.

### 🔹 Part 2: UCM (Use Case Manager)
ALSA controls are low-level ("Master Volume", "Switch 3").
Userspace needs high-level verbs ("Play Music", "Voice Call").
UCM is a set of text files that map Verbs to Mixer Settings.
*   `HiFi.conf`: Enable DAC, Enable HP Amp, Set Volume to 80%.

---

## 💻 Implementation: Jack Detection

> **Instruction:** Add a Headphone Jack to the Machine Driver.

### 👨‍💻 Code Implementation

#### Step 1: Structure Update
```c
struct my_machine_data {
    struct snd_soc_jack hp_jack;
    struct gpio_desc *hp_gpio;
};
```

#### Step 2: Pin Definition
```c
static struct snd_soc_jack_pin hp_jack_pins[] = {
    {
        .pin = "Headphone Jack", // Must match DAPM Widget name
        .mask = SND_JACK_HEADPHONE,
    },
};
```

#### Step 3: GPIO Definition
```c
static struct snd_soc_jack_gpio hp_jack_gpios[] = {
    {
        .name = "hp-detect",
        .report = SND_JACK_HEADPHONE,
        .debounce_time = 200,
        .invert = 0, // Active High
    },
};
```

#### Step 4: Initialization (in Machine Probe)
```c
// 1. Create Jack Object
ret = snd_soc_card_jack_new(card, "Headphone Jack", 
                            SND_JACK_HEADPHONE, &priv->hp_jack,
                            hp_jack_pins, ARRAY_SIZE(hp_jack_pins));

// 2. Link to GPIO
// Assuming we got the GPIO index/desc from DT or lookup
hp_jack_gpios[0].gpio = desc_to_gpio(priv->hp_gpio); 

ret = snd_soc_jack_add_gpios(&priv->hp_jack, 
                             ARRAY_SIZE(hp_jack_gpios), hp_jack_gpios);
```

---

## 🔬 Lab Exercise: Lab 168.1 - Testing Jack Events

### 1. Lab Objectives
- Load driver.
- Simulate GPIO toggle.
- Observe event.

### 2. Step-by-Step Guide
1.  **Find Input Device:**
    ```bash
    cat /proc/bus/input/devices
    # Look for "My Sound Card Headphone Jack"
    ```
2.  **Monitor:**
    ```bash
    evtest /dev/input/eventX
    ```
3.  **Toggle:** (Simulate via QEMU monitor or physical button).
    *   Output: `Event: type 5 (EV_SW), code 2 (SW_HEADPHONE_INSERT), value 1`

---

## 🧪 Additional / Advanced Labs

### Lab 2: DAPM Auto-Switching
- **Goal:** Automatically mute Speaker when HP is inserted.
- **Task:**
    1.  Ensure "Headphone Jack" pin is defined.
    2.  Ensure DAPM routes exist: "Headphone Jack" <- "HP Amp".
    3.  ASoC automatically powers down the Speaker path if it's not connected to an active pin? No, usually userspace (UCM) handles policy, but DAPM handles power.

### Lab 3: Mic Detection
- **Goal:** Headset (Mic + HP) detection.
- **Task:**
    1.  Use `SND_JACK_HEADSET` (Headphone + Microphone).
    2.  Some codecs (e.g., TS3A227E) have dedicated chips for this.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. No Event
*   **Cause:** GPIO polarity inverted.
*   **Cause:** Debounce time too long?

#### 2. "Headphone Jack" not found in DAPM
*   **Cause:** The string passed to `snd_soc_card_jack_new` MUST match a `SND_SOC_DAPM_HP` or `SND_SOC_DAPM_PIN` widget in the machine driver or codec driver.

---

## ⚡ Optimization & Best Practices

### UCM Profiles
*   Don't hardcode mixer settings in your application.
*   Write a UCM profile (`/usr/share/alsa/ucm2/MyCard/HiFi.conf`).
*   This allows PulseAudio to "Just Work".

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `SND_JACK_BTN_0`?
    *   **A:** Headset button (Play/Pause). Jack detection can also report button presses!
2.  **Q:** Why do we need `snd_soc_jack_add_gpios`?
    *   **A:** It's a helper that creates a `gpio_keys` style interrupt handler for you. You don't need to write your own ISR.

### Challenge Task
> **Task:** "The Button Press".
> *   Add a second GPIO for a button.
> *   Report `SND_JACK_BTN_0`.
> *   Verify `evtest` shows the button press.

---

## 📚 Further Reading & References
- [Kernel Documentation: sound/soc/jack.rst](https://www.kernel.org/doc/html/latest/sound/soc/jack.html)
- [ALSA UCM Documentation](https://github.com/alsa-project/alsa-ucm-conf)

---
