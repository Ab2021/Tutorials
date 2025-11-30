# Day 166: ALSA SoC (ASoC) Framework
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
1.  **Explain** the ASoC architecture (Machine, Platform, Codec).
2.  **Differentiate** between I2S (Digital Audio) and I2C (Control) interfaces.
3.  **Implement** a Dummy Codec Driver (`snd_soc_component_driver`).
4.  **Implement** a Simple Machine Driver (`snd_soc_card`).
5.  **Visualize** the DAPM (Dynamic Audio Power Management) graph.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 163 (ALSA).
    *   I2S Protocol.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why ASoC?
Standard ALSA drivers are monolithic (one driver for the whole card).
Embedded systems are modular:
*   **SoC:** Has an I2S controller (DMA).
*   **Codec:** External chip (Wolfson, Realtek) connected via I2S and I2C.
*   **Board:** Wires them together.

ASoC splits the driver into 3 parts:
1.  **Codec Driver:** Controls the external chip (Vol, Mute, DAPM). Reusable.
2.  **Platform Driver:** Controls the SoC I2S/DMA. Reusable.
3.  **Machine Driver:** The "Glue". Says "I2S0 is connected to WM8960". Board-specific.

### 🔹 Part 2: DAPM (Dynamic Audio Power Management)
ASoC automatically powers down parts of the codec that are not in use.
*   **Widgets:** ADC, DAC, Mixer, Pin (Headphone Jack).
*   **Routes:** Connections between widgets.
*   If "Headphone" is playing, power up DAC -> Mixer -> HP Amp.

---

## 💻 Implementation: Dummy Codec Driver

> **Instruction:** Create a reusable codec driver.

### 👨‍💻 Code Implementation

#### Step 1: Component Driver
```c
#include <sound/soc.h>

static const struct snd_kcontrol_new my_codec_controls[] = {
    SOC_SINGLE("Master Volume", 0x00, 0, 100, 0),
};

static const struct snd_soc_dapm_widget my_dapm_widgets[] = {
    SND_SOC_DAPM_OUTPUT("Speaker"),
    SND_SOC_DAPM_DAC("DAC", "Playback", SND_SOC_NOPM, 0, 0),
};

static const struct snd_soc_dapm_route my_dapm_routes[] = {
    { "Speaker", NULL, "DAC" },
};

static struct snd_soc_component_driver my_component_driver = {
    .controls = my_codec_controls,
    .num_controls = ARRAY_SIZE(my_codec_controls),
    .dapm_widgets = my_dapm_widgets,
    .num_dapm_widgets = ARRAY_SIZE(my_dapm_widgets),
    .dapm_routes = my_dapm_routes,
    .num_dapm_routes = ARRAY_SIZE(my_dapm_routes),
};
```

#### Step 2: DAI (Digital Audio Interface)
Describes the I2S capabilities.
```c
static struct snd_soc_dai_driver my_dai = {
    .name = "my-codec-dai",
    .playback = {
        .stream_name = "Playback",
        .channels_min = 2,
        .channels_max = 2,
        .rates = SNDRV_PCM_RATE_48000,
        .formats = SNDRV_PCM_FMTBIT_S16_LE,
    },
};
```

#### Step 3: Registration (Platform Driver)
```c
static int my_codec_probe(struct platform_device *pdev) {
    return devm_snd_soc_register_component(&pdev->dev,
                                           &my_component_driver,
                                           &my_dai, 1);
}
```

---

## 💻 Implementation: Machine Driver

> **Instruction:** Glue the Dummy Codec to a Dummy Platform.

### 👨‍💻 Code Implementation

#### Step 1: DAI Link
```c
static struct snd_soc_dai_link my_dai_link = {
    .name = "My Audio Link",
    .stream_name = "My Audio Stream",
    .codec_name = "my-codec.0", // Name of the platform_device + ID
    .codec_dai_name = "my-codec-dai",
    .cpu_dai_name = "snd-soc-dummy-dai", // Built-in dummy
    .platform_name = "snd-soc-dummy",
    .dai_fmt = SND_SOC_DAIFMT_I2S | SND_SOC_DAIFMT_NB_NF | SND_SOC_DAIFMT_CBS_CFS,
};
```

#### Step 2: Card
```c
static struct snd_soc_card my_card = {
    .name = "My Sound Card",
    .owner = THIS_MODULE,
    .dai_link = &my_dai_link,
    .num_links = 1,
};
```

#### Step 3: Registration
```c
static int my_machine_probe(struct platform_device *pdev) {
    my_card.dev = &pdev->dev;
    return devm_snd_soc_register_card(&pdev->dev, &my_card);
}
```

---

## 🔬 Lab Exercise: Lab 166.1 - ASoC Registration

### 1. Lab Objectives
- Compile Codec and Machine drivers.
- Load `snd-soc-dummy` (Kernel module).
- Load Codec.
- Load Machine.
- Verify Card.

### 2. Step-by-Step Guide
1.  **Load Dummy:** `modprobe snd-soc-dummy`.
2.  **Load Codec:** `insmod my_codec.ko`.
3.  **Load Machine:** `insmod my_machine.ko`.
4.  **Check:** `aplay -l`.
    *   Card: "My Sound Card".

---

## 🧪 Additional / Advanced Labs

### Lab 2: DAPM Graph
- **Goal:** Visualize power management.
- **Task:**
    1.  Play audio.
    2.  Check `/sys/kernel/debug/asoc/My Sound Card/dapm/`.
    3.  Read `bias_level` of the codec.

### Lab 3: I2C Codec
- **Goal:** Real world scenario.
- **Task:**
    1.  Change Codec Driver to be an I2C Driver (`i2c_driver`).
    2.  Use `regmap` for controls.
    3.  In Machine Driver, refer to codec by I2C name ("1-001a").

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No DAI link found"
*   **Cause:** Mismatch in names (`codec_name`, `codec_dai_name`).
*   **Fix:** Check `/sys/kernel/debug/asoc/dais` to see registered DAIs.

#### 2. "ASoC: Failed to add route"
*   **Cause:** Route refers to a widget that doesn't exist.
*   **Fix:** Check spelling in `dapm_widgets` and `dapm_routes`.

---

## ⚡ Optimization & Best Practices

### Device Tree
*   Modern Machine Drivers parse Device Tree.
*   `audio-graph-card` is a generic machine driver that builds the card from DT nodes, removing the need for C code in simple cases.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a DAI?
    *   **A:** Digital Audio Interface. It defines the physical bus (I2S, PCM, AC97) and its parameters (Rate, Format).
2.  **Q:** Why do we need a Machine Driver?
    *   **A:** Because the Codec doesn't know which CPU it's connected to, and the CPU doesn't know which Codec is attached. The Machine Driver (or Device Tree) provides this binding.

### Challenge Task
> **Task:** "The Mic Path".
> *   Add a "Mic" Input Widget and a "Capture" Stream to the Codec.
> *   Add a route "ADC" <- "Mic".
> *   Verify you can record.

---

## 📚 Further Reading & References
- [Kernel Documentation: sound/soc/index.rst](https://www.kernel.org/doc/html/latest/sound/soc/index.html)

---
