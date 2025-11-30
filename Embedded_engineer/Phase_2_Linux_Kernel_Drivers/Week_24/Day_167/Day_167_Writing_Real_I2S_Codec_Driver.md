# Day 167: Writing a Real I2S Codec Driver
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
1.  **Develop** an I2C-controlled Codec Driver (e.g., SGTL5000 or WM8960 style).
2.  **Implement** `hw_params` to configure sample rate and bit depth.
3.  **Manage** System Clock (MCLK) and Bit Clock (BCLK) ratios.
4.  **Use** Regmap for register access and caching.
5.  **Debug** "No Sound" issues caused by clock mismatches.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 166 (ASoC).
    *   Day 146 (Regmap).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The I2S Bus
*   **BCLK (Bit Clock):** Toggles for every bit. `Rate * Channels * Bits`.
*   **LRCLK (Left/Right Clock):** Toggles for every frame (Sample Rate).
*   **DIN/DOUT:** Data lines.
*   **MCLK (Master Clock):** High speed clock (e.g., 12.288 MHz) used by the codec to drive its internal DSP/DAC.

### 🔹 Part 2: Clocking Modes
*   **Master Mode:** Codec generates BCLK/LRCLK from MCLK.
*   **Slave Mode:** SoC generates BCLK/LRCLK. Codec just listens. (Most common in simple setups).

---

## 💻 Implementation: The Codec Driver

> **Instruction:** We will simulate a codec with registers.

### 👨‍💻 Code Implementation

#### Step 1: Regmap Config
```c
static const struct regmap_config my_regmap = {
    .reg_bits = 8,
    .val_bits = 8,
    .max_register = 0xFF,
    .cache_type = REGCACHE_RBTREE,
};
```

#### Step 2: DAI Operations (`hw_params`)
Called when a stream starts. We must configure the codec for the requested rate.
```c
static int my_hw_params(struct snd_pcm_substream *substream,
                        struct snd_pcm_hw_params *params,
                        struct snd_soc_dai *dai) {
    struct snd_soc_component *component = dai->component;
    int rate = params_rate(params);
    int width = params_width(params);
    u8 val = 0;

    // 1. Set Sample Rate
    switch (rate) {
    case 48000: val = 0x01; break;
    case 44100: val = 0x02; break;
    default: return -EINVAL;
    }
    snd_soc_component_update_bits(component, REG_RATE, 0x0F, val);

    // 2. Set Bit Width
    switch (width) {
    case 16: val = 0x10; break;
    case 24: val = 0x20; break;
    default: return -EINVAL;
    }
    snd_soc_component_update_bits(component, REG_FORMAT, 0xF0, val);

    return 0;
}

static const struct snd_soc_dai_ops my_dai_ops = {
    .hw_params = my_hw_params,
};
```

#### Step 3: DAI Driver
```c
static struct snd_soc_dai_driver my_dai = {
    .name = "my-i2s-hifi",
    .playback = {
        .stream_name = "Playback",
        .channels_min = 2,
        .channels_max = 2,
        .rates = SNDRV_PCM_RATE_44100 | SNDRV_PCM_RATE_48000,
        .formats = SNDRV_PCM_FMTBIT_S16_LE | SNDRV_PCM_FMTBIT_S24_LE,
    },
    .ops = &my_dai_ops,
};
```

#### Step 4: I2C Probe
```c
static int my_i2c_probe(struct i2c_client *i2c, const struct i2c_device_id *id) {
    struct regmap *map;
    
    map = devm_regmap_init_i2c(i2c, &my_regmap);
    if (IS_ERR(map)) return PTR_ERR(map);
    
    return devm_snd_soc_register_component(&i2c->dev,
                                           &my_component_driver,
                                           &my_dai, 1);
}
```

---

## 🔬 Lab Exercise: Lab 167.1 - Clock Configuration

### 1. Lab Objectives
- Implement `set_sysclk` DAI op.
- Configure MCLK in Machine Driver.

### 2. Step-by-Step Guide
1.  **Codec Op:**
    ```c
    static int my_set_sysclk(struct snd_soc_dai *dai, int clk_id, unsigned int freq, int dir) {
        // Store freq for later calculation in hw_params
        return 0;
    }
    ```
2.  **Machine Driver:**
    ```c
    // In hw_params of machine driver
    snd_soc_dai_set_sysclk(codec_dai, 0, 12288000, SND_SOC_CLOCK_IN);
    ```

---

## 🧪 Additional / Advanced Labs

### Lab 2: Digital Mute
- **Goal:** Implement `digital_mute` op.
- **Task:**
    1.  Add `.digital_mute = my_mute` to `dai_ops`.
    2.  If `mute` is true, write to Mute Register.
    3.  ASoC calls this automatically when stopping playback.

### Lab 3: PLL Configuration
- **Goal:** Handle non-standard MCLK.
- **Task:**
    1.  Implement `set_pll`.
    2.  Calculate dividers to generate 12.288MHz from 24MHz input.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Audio plays fast/slow
*   **Cause:** Mismatch between SoC I2S clock and Codec expectation.
*   **Fix:** Ensure `hw_params` sets the correct dividers.

#### 2. "Pop" noise on start
*   **Cause:** DC offset jumping.
*   **Fix:** Use DAPM to ramp up volume or enable "Soft Mute" features in the codec.

---

## ⚡ Optimization & Best Practices

### `SND_SOC_DAIFMT_`
*   Be very careful with `SND_SOC_DAIFMT_CBS_CFS` (Codec Bit Slave, Frame Slave).
*   This means the Codec is SLAVE. The SoC must be MASTER.
*   If both are Slave, no clock = no sound.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `hw_params` and `prepare`?
    *   **A:** `hw_params` allocates buffers and sets formats (heavy). `prepare` resets pointers and clears FIFOs (light).
2.  **Q:** Why do we need `regmap`?
    *   **A:** To cache register values. If we mute/unmute, we don't want to read the register over I2C every time if we know the value.

### Challenge Task
> **Task:** "The Format Detector".
> *   In `hw_params`, print the calculated BCLK frequency.
> *   `BCLK = Rate * Channels * Width`.
> *   Verify it matches 1.536 MHz for 48kHz/16-bit/Stereo.

---

## 📚 Further Reading & References
- [Kernel Documentation: sound/soc/codec.rst](https://www.kernel.org/doc/html/latest/sound/soc/codec.html)

---
