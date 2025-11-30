# Day 161: The ISP (Image Signal Processor)
## Phase 2: Linux Kernel & Device Drivers | Week 23: V4L2 Subdevices

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
1.  **Explain** the function of an ISP (Debayering, AWB, AE, Gamma).
2.  **Architect** an ISP driver as a V4L2 Subdevice with Sink and Source pads.
3.  **Implement** format propagation (Sink Pad Format -> Source Pad Format).
4.  **Simulate** simple image processing (e.g., Grayscale conversion) in a virtual driver.
5.  **Configure** the ISP pipeline using `media-ctl`.

---

## 📚 Prerequisites & Preparation
*   **Hardware Required:**
    *   PC with Linux (or QEMU).
*   **Software Required:**
    *   Previous Day's Driver environment.
*   **Prior Knowledge:**
    *   Day 156 (Subdevices).
    *   Bayer Pattern.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What does an ISP do?
Raw sensors output a Bayer Pattern (Mosaic). It looks green and checkerboard-like.
The ISP (Image Signal Processor) converts this to a viewable image (RGB/YUV).
*   **Demosaicing (Debayering):** Interpolating colors.
*   **Auto White Balance (AWB):** Adjusting gains so white looks white.
*   **Auto Exposure (AE):** Adjusting sensor exposure time.
*   **Lens Shading Correction:** Fixing dark corners.

### 🔹 Part 2: The ISP as a Subdevice
In V4L2, the ISP is just another block in the graph.
*   **Sink Pad:** Receives Raw Bayer (e.g., `SRGGB10`).
*   **Source Pad:** Outputs YUV/RGB (e.g., `YUYV`).
*   **Processing:** Done in hardware (or software for our simulation).

---

## 💻 Implementation: The Virtual ISP Driver

> **Instruction:** Create a subdevice that accepts Bayer and outputs YUYV.

### 👨‍💻 Code Implementation

#### Step 1: Structure
```c
struct my_isp {
    struct v4l2_subdev sd;
    struct media_pad pads[2]; // 0: Sink, 1: Source
    struct v4l2_mbus_framefmt sink_fmt;
    struct v4l2_mbus_framefmt src_fmt;
};
```

#### Step 2: Pad Init (in Probe)
```c
sensor->pads[0].flags = MEDIA_PAD_FL_SINK;
sensor->pads[1].flags = MEDIA_PAD_FL_SOURCE;
sensor->sd.entity.function = MEDIA_ENT_F_PROC_VIDEO_ISP;

ret = media_entity_pads_init(&sensor->sd.entity, 2, sensor->pads);
```

#### Step 3: Format Propagation (set_fmt)
If the user changes the Sink format (Input), the Source format (Output) might need to change.
```c
static int isp_set_fmt(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                       struct v4l2_subdev_format *fmt) {
    struct my_isp *isp = to_my_isp(sd);
    
    if (fmt->pad == 0) { // Sink
        // 1. Accept the format (e.g., SRGGB10)
        isp->sink_fmt = fmt->format;
        
        // 2. Reset Source format to match dimensions, but force YUYV
        isp->src_fmt = fmt->format;
        isp->src_fmt.code = MEDIA_BUS_FMT_YUYV8_2X8;
    } else { // Source
        // 1. User wants to change output format
        // 2. Validate: Can we convert Sink -> Requested Source?
        if (fmt->format.code != MEDIA_BUS_FMT_YUYV8_2X8)
             fmt->format.code = MEDIA_BUS_FMT_YUYV8_2X8; // Force YUYV
             
        isp->src_fmt = fmt->format;
    }
    return 0;
}
```

#### Step 4: Enum MBUS Code
Tell userspace what we support.
```c
static int isp_enum_mbus_code(struct v4l2_subdev *sd, struct v4l2_subdev_state *state,
                              struct v4l2_subdev_mbus_code_enum *code) {
    if (code->pad == 0) { // Sink
        if (code->index > 0) return -EINVAL;
        code->code = MEDIA_BUS_FMT_SRGGB10_1X10;
    } else { // Source
        if (code->index > 0) return -EINVAL;
        code->code = MEDIA_BUS_FMT_YUYV8_2X8;
    }
    return 0;
}
```

---

## 🔬 Lab Exercise: Lab 161.1 - Configuring the ISP

### 1. Lab Objectives
- Load Sensor, ISP, and Bridge.
- Link them: Sensor -> ISP -> Bridge.
- Configure formats.

### 2. Step-by-Step Guide
1.  **Link:**
    ```bash
    media-ctl -l '"Sensor":0 -> "ISP":0[1]'
    media-ctl -l '"ISP":1 -> "Bridge":0[1]'
    ```
2.  **Configure Sensor Output:**
    ```bash
    media-ctl -V '"Sensor":0 [fmt:SRGGB10/1920x1080]'
    ```
3.  **Configure ISP Input:**
    ```bash
    media-ctl -V '"ISP":0 [fmt:SRGGB10/1920x1080]'
    ```
4.  **Check ISP Output:**
    ```bash
    media-ctl -p
    # ISP:1 Source [fmt:YUYV8_2X8/1920x1080]
    ```
    *   It should automatically match the resolution but change the code to YUYV.

---

## 🧪 Additional / Advanced Labs

### Lab 2: Software Debayering (Slow)
- **Goal:** Implement a simple Nearest Neighbor debayering in the Bridge driver (since the ISP driver is just a config shell in this simulation).
- **Task:**
    1.  In Bridge `buf_queue`, if input is Bayer, run a loop to convert to RGB.
    2.  `R = Pixel[x,y]`, `G = (Pixel[x-1,y] + Pixel[x+1,y])/2`, etc.

### Lab 3: Passthrough Mode
- **Goal:** Allow Raw Bayer to pass through the ISP unmodified.
- **Task:**
    1.  Add a control `V4L2_CID_BYPASS`.
    2.  If enabled, `set_fmt` on Source Pad allows `SRGGB10`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Link validation failed"
*   **Cause:** You configured Sensor to 1080p, but ISP Sink is still default (640x480).
*   **Fix:** You must configure the pipeline from Source to Sink order.
    1.  Sensor Out
    2.  ISP In (Propagates to ISP Out)
    3.  Bridge In

---

## ⚡ Optimization & Best Practices

### Hardware ISPs
*   Real ISPs (Rockchip RKISP1, Raspberry Pi ISP) are extremely complex.
*   They use `v4l2_params` buffers to accept tuning parameters (Gamma tables, Matrix coefficients) from userspace (libcamera).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does the ISP have two pads?
    *   **A:** One for Input (Sink) and one for Output (Source). It transforms the data stream.
2.  **Q:** What happens if I change the Sensor resolution?
    *   **A:** Ideally, the ISP driver should detect this change (via `link_validate` or notification) and update its Sink format, or return an error if the config is stale.

### Challenge Task
> **Task:** "The Downscaler".
> *   Modify the ISP `set_fmt` logic.
> *   If the user requests a smaller resolution on the Source Pad than the Sink Pad, accept it (Scaling).
> *   If the user requests a larger resolution, reject it (Upscaling not supported).

---

## 📚 Further Reading & References
- [Kernel Documentation: driver-api/media/v4l2-subdev.rst](https://www.kernel.org/doc/html/latest/driver-api/media/v4l2-subdev.html)

---
