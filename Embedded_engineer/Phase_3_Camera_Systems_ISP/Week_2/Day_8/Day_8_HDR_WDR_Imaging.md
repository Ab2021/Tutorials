# Day 8: HDR and WDR Imaging Techniques
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** HDR and WDR imaging principles and differences
2. **Implement** multi-exposure HDR capture and fusion
3. **Configure** sensor WDR modes (split pixel, multiple exposure)
4. **Develop** tone mapping algorithms for HDR display
5. **Debug** HDR artifacts (ghosting, halos, noise)
6. **Optimize** real-time HDR processing performance

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera sensor with HDR/WDR support
*   **Software:** Image processing libraries, OpenCV
*   **Knowledge:** Exposure control, tone mapping, image fusion
*   **Tools:** HDR test scenes, photometric measurement tools

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: HDR vs WDR Fundamentals

#### 1.1 Dynamic Range Concepts

**Dynamic Range Definition:**

```
DR (dB) = 20 × log10(Signal_max / Signal_min)

For image sensor:
DR = 20 × log10(Full_Well_Capacity / Read_Noise)

Example:
FWC = 20,000 electrons
Read Noise = 2 electrons
DR = 20 × log10(20000/2) = 80 dB
```

**Scene Dynamic Range:**

```
Typical scenes:
- Indoor: 60-80 dB
- Outdoor daylight: 80-100 dB
- Backlit scenes: 100-120 dB
- Sunset/sunrise: 120-140 dB

Single exposure sensor: 60-80 dB
HDR techniques: 100-140 dB
```

#### 1.2 HDR vs WDR

**High Dynamic Range (HDR):**
- Multiple exposures combined
- Temporal approach
- Best quality, potential motion artifacts
- Post-processing intensive

**Wide Dynamic Range (WDR):**
- Single exposure with special sensor modes
- Spatial approach (split pixel, dual conversion gain)
- Real-time capable
- Limited DR extension (~20-30 dB)

**Comparison:**

```c
/**
 * @brief HDR/WDR characteristics
 */
struct dr_technique {
    const char *name;
    unsigned int dr_db;
    bool real_time;
    bool motion_artifacts;
    unsigned int exposures;
};

static const struct dr_technique techniques[] = {
    {"Standard", 70, true, false, 1},
    {"WDR (Split Pixel)", 90, true, false, 1},
    {"WDR (DCG)", 95, true, false, 1},
    {"HDR (2 Exposure)", 100, false, true, 2},
    {"HDR (3 Exposure)", 110, false, true, 3},
    {"HDR (4 Exposure)", 120, false, true, 4},
};
```

### 🔹 Part 2: Multi-Exposure HDR

#### 2.1 Exposure Bracketing

**Optimal Exposure Spacing:**

```c
/**
 * @brief Calculate optimal exposure sequence for HDR
 */
struct exposure_sequence {
    unsigned int num_exposures;
    float *exposure_times;  /* In seconds */
    float *gains;
};

static struct exposure_sequence calculate_hdr_sequence(
    float scene_dr_stops,
    float sensor_dr_stops,
    float base_exposure)
{
    struct exposure_sequence seq = {0};
    
    /* Number of exposures needed */
    seq.num_exposures = (unsigned int)ceilf(scene_dr_stops / sensor_dr_stops) + 1;
    
    seq.exposure_times = malloc(seq.num_exposures * sizeof(float));
    seq.gains = malloc(seq.num_exposures * sizeof(float));
    
    /* Calculate exposure spacing (typically 2-3 stops apart) */
    float spacing_stops = 2.0f;
    
    for (unsigned int i = 0; i < seq.num_exposures; i++) {
        /* Exposure in stops relative to base */
        float stops = -scene_dr_stops/2 + i * spacing_stops;
        
        seq.exposure_times[i] = base_exposure * powf(2.0f, stops);
        seq.gains[i] = 1.0f;  /* Keep gain constant */
    }
    
    return seq;
}

/* Example: Scene DR = 12 stops, Sensor DR = 6 stops, Base = 10ms */
/* Exposures: 2.5ms, 10ms, 40ms (3 exposures, 2 stops apart) */
```

#### 2.2 HDR Capture

**Sequential Capture:**

```c
/**
 * @brief Capture HDR image sequence
 */
struct hdr_capture {
    unsigned int num_images;
    uint16_t **images;
    float *exposures;
    unsigned int width;
    unsigned int height;
};

static struct hdr_capture *capture_hdr_sequence(
    struct camera_manager *cam,
    struct exposure_sequence *seq)
{
    struct hdr_capture *hdr = calloc(1, sizeof(*hdr));
    hdr->num_images = seq->num_exposures;
    hdr->width = cam->config.width;
    hdr->height = cam->config.height;
    
    hdr->images = malloc(hdr->num_images * sizeof(uint16_t*));
    hdr->exposures = malloc(hdr->num_images * sizeof(float));
    
    for (unsigned int i = 0; i < hdr->num_images; i++) {
        /* Set exposure */
        camera_set_exposure(cam, seq->exposure_times[i]);
        
        /* Wait for exposure to take effect */
        usleep(50000);  /* 50ms */
        
        /* Capture frame */
        hdr->images[i] = malloc(hdr->width * hdr->height * sizeof(uint16_t));
        camera_capture_frame(cam, hdr->images[i]);
        
        hdr->exposures[i] = seq->exposure_times[i];
        
        printf("Captured exposure %u: %.3f ms\n", i, 
               seq->exposure_times[i] * 1000);
    }
    
    return hdr;
}
```

#### 2.3 HDR Fusion Algorithms

**Debevec-Malik Algorithm:**

```c
/**
 * @brief HDR fusion using Debevec-Malik algorithm
 */

/* Camera response function (CRF) - simplified linear */
static float camera_response(uint16_t pixel_value, float max_value)
{
    return (float)pixel_value / max_value;
}

/* Weight function - hat function favoring middle values */
static float weight_function(uint16_t pixel_value, uint16_t max_value)
{
    float normalized = (float)pixel_value / max_value;
    
    if (normalized <= 0.5f) {
        return normalized * 2.0f;
    } else {
        return (1.0f - normalized) * 2.0f;
    }
}

/**
 * @brief Merge HDR exposures
 */
static void hdr_merge(
    struct hdr_capture *hdr,
    float *output_hdr)  /* Output: linear HDR values */
{
    const uint16_t max_val = 1023;  /* 10-bit */
    
    for (unsigned int y = 0; y < hdr->height; y++) {
        for (unsigned int x = 0; x < hdr->width; x++) {
            unsigned int idx = y * hdr->width + x;
            
            float sum_weighted_radiance = 0.0f;
            float sum_weights = 0.0f;
            
            for (unsigned int i = 0; i < hdr->num_images; i++) {
                uint16_t pixel = hdr->images[i][idx];
                
                /* Skip saturated or too dark pixels */
                if (pixel < 10 || pixel > max_val - 10)
                    continue;
                
                /* Weight based on pixel value */
                float w = weight_function(pixel, max_val);
                
                /* Recover scene radiance */
                float response = camera_response(pixel, max_val);
                float radiance = response / hdr->exposures[i];
                
                sum_weighted_radiance += w * radiance;
                sum_weights += w;
            }
            
            if (sum_weights > 0.0f) {
                output_hdr[idx] = sum_weighted_radiance / sum_weights;
            } else {
                output_hdr[idx] = 0.0f;
            }
        }
    }
}
```

**Mertens Exposure Fusion (No HDR intermediate):**

```c
/**
 * @brief Mertens exposure fusion - direct LDR output
 */

static float quality_measure(
    uint16_t pixel,
    uint16_t max_val,
    float contrast,
    float saturation,
    float well_exposedness)
{
    /* Well-exposedness: Gaussian centered at 0.5 */
    float normalized = (float)pixel / max_val;
    float we = expf(-powf(normalized - 0.5f, 2.0f) / (2.0f * 0.2f * 0.2f));
    
    /* Combine quality measures */
    return powf(contrast, 1.0f) * 
           powf(saturation, 1.0f) * 
           powf(we, well_exposedness);
}

static void mertens_fusion(
    struct hdr_capture *hdr,
    uint16_t *output_ldr)
{
    const uint16_t max_val = 1023;
    
    /* Build quality maps */
    float **quality_maps = malloc(hdr->num_images * sizeof(float*));
    for (unsigned int i = 0; i < hdr->num_images; i++) {
        quality_maps[i] = malloc(hdr->width * hdr->height * sizeof(float));
        
        for (unsigned int p = 0; p < hdr->width * hdr->height; p++) {
            /* Calculate local contrast (simplified) */
            float contrast = 1.0f;
            
            /* Calculate saturation (for color images) */
            float saturation = 1.0f;
            
            /* Well-exposedness */
            quality_maps[i][p] = quality_measure(
                hdr->images[i][p], max_val,
                contrast, saturation, 1.0f
            );
        }
    }
    
    /* Normalize quality maps */
    for (unsigned int p = 0; p < hdr->width * hdr->height; p++) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < hdr->num_images; i++) {
            sum += quality_maps[i][p];
        }
        
        if (sum > 0.0f) {
            for (unsigned int i = 0; i < hdr->num_images; i++) {
                quality_maps[i][p] /= sum;
            }
        }
    }
    
    /* Weighted blend */
    for (unsigned int p = 0; p < hdr->width * hdr->height; p++) {
        float blended = 0.0f;
        
        for (unsigned int i = 0; i < hdr->num_images; i++) {
            blended += quality_maps[i][p] * hdr->images[i][p];
        }
        
        output_ldr[p] = (uint16_t)blended;
    }
    
    /* Cleanup */
    for (unsigned int i = 0; i < hdr->num_images; i++) {
        free(quality_maps[i]);
    }
    free(quality_maps);
}
```

### 🔹 Part 3: Tone Mapping

#### 3.1 Global Tone Mapping

**Reinhard Operator:**

```c
/**
 * @brief Reinhard global tone mapping
 */
static void tonemap_reinhard_global(
    float *hdr,
    uint16_t *ldr,
    unsigned int width,
    unsigned int height,
    float key_value)
{
    unsigned int total_pixels = width * height;
    
    /* Calculate log-average luminance */
    float log_avg = 0.0f;
    const float delta = 1e-6f;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        log_avg += logf(hdr[i] + delta);
    }
    log_avg = expf(log_avg / total_pixels);
    
    /* Scale HDR values */
    float scale = key_value / log_avg;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        float scaled = hdr[i] * scale;
        
        /* Reinhard operator: L_d = L / (1 + L) */
        float mapped = scaled / (1.0f + scaled);
        
        /* Convert to LDR range */
        ldr[i] = (uint16_t)(mapped * 1023);
    }
}
```

**Filmic Tone Mapping (ACES):**

```c
/**
 * @brief ACES filmic tone mapping
 */
static float aces_tonemap(float x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

static void tonemap_aces(
    float *hdr,
    uint16_t *ldr,
    unsigned int width,
    unsigned int height)
{
    unsigned int total_pixels = width * height;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        float mapped = aces_tonemap(hdr[i]);
        
        /* Clamp and convert */
        if (mapped < 0.0f) mapped = 0.0f;
        if (mapped > 1.0f) mapped = 1.0f;
        
        ldr[i] = (uint16_t)(mapped * 1023);
    }
}
```

#### 3.2 Local Tone Mapping

**Durand Bilateral Tone Mapping:**

```c
/**
 * @brief Durand bilateral tone mapping
 */
static void tonemap_durand(
    float *hdr,
    uint16_t *ldr,
    unsigned int width,
    unsigned int height)
{
    /* 1. Compute log luminance */
    float *log_lum = malloc(width * height * sizeof(float));
    for (unsigned int i = 0; i < width * height; i++) {
        log_lum[i] = logf(hdr[i] + 1e-6f);
    }
    
    /* 2. Bilateral filter to get base layer */
    float *base = malloc(width * height * sizeof(float));
    bilateral_filter_float(log_lum, base, width, height, 5.0f, 0.4f);
    
    /* 3. Compute detail layer */
    float *detail = malloc(width * height * sizeof(float));
    for (unsigned int i = 0; i < width * height; i++) {
        detail[i] = log_lum[i] - base[i];
    }
    
    /* 4. Compress base layer */
    float base_min = FLT_MAX, base_max = -FLT_MAX;
    for (unsigned int i = 0; i < width * height; i++) {
        if (base[i] < base_min) base_min = base[i];
        if (base[i] > base_max) base_max = base[i];
    }
    
    float compression_factor = 5.0f;  /* Target DR in log space */
    float scale = compression_factor / (base_max - base_min);
    
    for (unsigned int i = 0; i < width * height; i++) {
        base[i] = (base[i] - base_min) * scale;
    }
    
    /* 5. Reconstruct and convert to LDR */
    for (unsigned int i = 0; i < width * height; i++) {
        float compressed_log = base[i] + detail[i];
        float linear = expf(compressed_log);
        
        ldr[i] = (uint16_t)(linear * 1023);
        if (ldr[i] > 1023) ldr[i] = 1023;
    }
    
    free(log_lum);
    free(base);
    free(detail);
}
```

### 🔹 Part 4: Sensor WDR Modes

#### 4.1 Split Pixel WDR

**Concept:**
- Each pixel has two photodiodes with different sensitivities
- High sensitivity for shadows
- Low sensitivity for highlights
- Combined in single exposure

```c
/**
 * @brief Process split-pixel WDR sensor output
 */
struct split_pixel_config {
    float high_gain;   /* High sensitivity gain */
    float low_gain;    /* Low sensitivity gain */
    uint16_t threshold; /* Switching threshold */
};

static void process_split_pixel_wdr(
    uint16_t *raw_high,  /* High sensitivity pixels */
    uint16_t *raw_low,   /* Low sensitivity pixels */
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    struct split_pixel_config *config)
{
    for (unsigned int i = 0; i < width * height; i++) {
        uint16_t high = raw_high[i];
        uint16_t low = raw_low[i];
        
        if (high < config->threshold) {
            /* Use high sensitivity pixel */
            output[i] = high;
        } else {
            /* Use low sensitivity pixel, scaled */
            float scaled = low * (config->high_gain / config->low_gain);
            output[i] = (uint16_t)scaled;
            if (output[i] > 1023) output[i] = 1023;
        }
    }
}
```

#### 4.2 Dual Conversion Gain (DCG)

**Concept:**
- Single photodiode, two readout paths
- High conversion gain for low light
- Low conversion gain for bright light
- Switched based on signal level

```c
/**
 * @brief Configure sensor DCG mode
 */
static int configure_dcg_mode(
    struct camera_manager *cam,
    bool enable)
{
    /* Sensor-specific register configuration */
    /* Example for hypothetical sensor */
    
    if (enable) {
        /* Enable DCG mode */
        sensor_write_reg(cam, 0x3100, 0x01);
        
        /* Set switching threshold */
        sensor_write_reg(cam, 0x3101, 0x80);  /* Mid-range */
        
        /* Set gain ratio */
        sensor_write_reg(cam, 0x3102, 0x04);  /* 4x ratio */
    } else {
        /* Disable DCG */
        sensor_write_reg(cam, 0x3100, 0x00);
    }
    
    return 0;
}
```

#### 4.3 Multiple Exposure WDR

**Staggered HDR:**
- Alternating long/short exposures
- Even rows: long exposure
- Odd rows: short exposure
- Single frame time

```c
/**
 * @brief Process staggered HDR sensor output
 */
static void process_staggered_hdr(
    uint16_t *raw,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float exposure_ratio)
{
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            
            if (y % 2 == 0) {
                /* Long exposure row */
                if (raw[idx] < 900) {  /* Not saturated */
                    output[idx] = raw[idx];
                } else {
                    /* Use short exposure from adjacent row */
                    if (y + 1 < height) {
                        unsigned int short_idx = (y + 1) * width + x;
                        output[idx] = (uint16_t)(raw[short_idx] * exposure_ratio);
                    }
                }
            } else {
                /* Short exposure row - interpolate from long */
                if (y > 0) {
                    unsigned int long_idx = (y - 1) * width + x;
                    output[idx] = output[long_idx];
                }
            }
        }
    }
}
```

---

## 💻 Implementation Examples

### Example 1: Complete HDR Capture and Processing

```c
/**
 * @file hdr_processor.c
 * @brief Complete HDR capture and processing pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

struct hdr_processor {
    struct camera_manager *camera;
    
    /* Configuration */
    unsigned int num_exposures;
    float *exposure_times;
    
    /* Buffers */
    struct hdr_capture *capture;
    float *hdr_buffer;
    uint16_t *ldr_output;
    
    /* Tone mapping */
    enum {
        TONEMAP_REINHARD,
        TONEMAP_ACES,
        TONEMAP_DURAND
    } tonemap_method;
};

/**
 * @brief Initialize HDR processor
 */
static struct hdr_processor *hdr_processor_init(
    struct camera_manager *cam,
    unsigned int num_exposures)
{
    struct hdr_processor *hdr = calloc(1, sizeof(*hdr));
    hdr->camera = cam;
    hdr->num_exposures = num_exposures;
    
    /* Calculate exposure sequence */
    hdr->exposure_times = malloc(num_exposures * sizeof(float));
    
    float base_exposure = 0.010f;  /* 10ms */
    for (unsigned int i = 0; i < num_exposures; i++) {
        /* 2 stops apart */
        float stops = -2.0f + i * 2.0f;
        hdr->exposure_times[i] = base_exposure * powf(2.0f, stops);
    }
    
    /* Allocate buffers */
    unsigned int pixels = cam->config.width * cam->config.height;
    hdr->hdr_buffer = malloc(pixels * sizeof(float));
    hdr->ldr_output = malloc(pixels * sizeof(uint16_t));
    
    hdr->tonemap_method = TONEMAP_ACES;
    
    return hdr;
}

/**
 * @brief Process HDR frame
 */
static int hdr_processor_process(struct hdr_processor *hdr)
{
    printf("HDR Processing:\n");
    
    /* 1. Capture exposure sequence */
    printf("  [1/3] Capturing %u exposures...\n", hdr->num_exposures);
    
    struct exposure_sequence seq = {
        .num_exposures = hdr->num_exposures,
        .exposure_times = hdr->exposure_times,
        .gains = NULL
    };
    
    hdr->capture = capture_hdr_sequence(hdr->camera, &seq);
    
    /* 2. Merge to HDR */
    printf("  [2/3] Merging exposures...\n");
    hdr_merge(hdr->capture, hdr->hdr_buffer);
    
    /* 3. Tone mapping */
    printf("  [3/3] Tone mapping...\n");
    
    switch (hdr->tonemap_method) {
    case TONEMAP_REINHARD:
        tonemap_reinhard_global(hdr->hdr_buffer, hdr->ldr_output,
                               hdr->capture->width, hdr->capture->height,
                               0.18f);
        break;
    
    case TONEMAP_ACES:
        tonemap_aces(hdr->hdr_buffer, hdr->ldr_output,
                    hdr->capture->width, hdr->capture->height);
        break;
    
    case TONEMAP_DURAND:
        tonemap_durand(hdr->hdr_buffer, hdr->ldr_output,
                      hdr->capture->width, hdr->capture->height);
        break;
    }
    
    printf("HDR processing complete\n");
    
    return 0;
}

/**
 * @brief Save HDR output
 */
static int hdr_save_output(
    struct hdr_processor *hdr,
    const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        return -1;
    }
    
    size_t pixels = hdr->capture->width * hdr->capture->height;
    fwrite(hdr->ldr_output, sizeof(uint16_t), pixels, f);
    
    fclose(f);
    
    printf("Saved HDR output to %s\n", filename);
    
    return 0;
}

/**
 * @brief Example usage
 */
int main(int argc, char **argv)
{
    /* Initialize camera */
    struct camera_config config = {
        .width = 1920,
        .height = 1080,
        .fps = 30,
        .pixelformat = V4L2_PIX_FMT_SRGGB10,
    };
    
    struct camera_manager *cam = camera_init("/dev/video0", &config);
    if (!cam) {
        fprintf(stderr, "Failed to initialize camera\n");
        return 1;
    }
    
    /* Initialize HDR processor */
    struct hdr_processor *hdr = hdr_processor_init(cam, 3);
    
    /* Process HDR image */
    hdr_processor_process(hdr);
    
    /* Save output */
    hdr_save_output(hdr, "hdr_output.raw");
    
    /* Cleanup */
    free(hdr->hdr_buffer);
    free(hdr->ldr_output);
    free(hdr->exposure_times);
    free(hdr);
    camera_cleanup(cam);
    
    return 0;
}
```

### Example 2: Real-Time WDR Processing

```c
/**
 * @file realtime_wdr.c
 * @brief Real-time WDR processing for sensor WDR modes
 */

struct wdr_processor {
    enum {
        WDR_SPLIT_PIXEL,
        WDR_DCG,
        WDR_STAGGERED
    } mode;
    
    struct split_pixel_config split_config;
    float exposure_ratio;
    
    /* Buffers */
    uint16_t *temp_buffer;
};

/**
 * @brief Process WDR frame in real-time
 */
static void wdr_process_frame(
    struct wdr_processor *wdr,
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height)
{
    switch (wdr->mode) {
    case WDR_SPLIT_PIXEL:
        /* Extract high/low sensitivity pixels */
        /* Sensor-specific demuxing */
        process_split_pixel_wdr(input, input + width*height/2,
                               output, width, height,
                               &wdr->split_config);
        break;
    
    case WDR_DCG:
        /* DCG processing (sensor does most work) */
        memcpy(output, input, width * height * sizeof(uint16_t));
        break;
    
    case WDR_STAGGERED:
        process_staggered_hdr(input, output, width, height,
                             wdr->exposure_ratio);
        break;
    }
}

/**
 * @brief Real-time WDR streaming
 */
static void *wdr_stream_thread(void *arg)
{
    struct wdr_processor *wdr = arg;
    
    while (running) {
        /* Get frame from camera */
        uint16_t *frame = camera_get_frame();
        
        /* Process WDR */
        wdr_process_frame(wdr, frame, output_buffer,
                         width, height);
        
        /* Display/encode output */
        display_frame(output_buffer);
    }
    
    return NULL;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: HDR Capture and Evaluation

**Objective:** Capture and evaluate HDR images.

**Procedure:**

```bash
#!/bin/bash
# hdr_capture_test.sh

echo "HDR Capture Test"
echo "================"

# Capture exposure sequence
for exp in 2.5 10 40; do
    echo "Capturing ${exp}ms exposure..."
    v4l2-ctl --set-ctrl=exposure_absolute=$((exp * 100))
    sleep 0.1
    v4l2-ctl --stream-mmap --stream-to=exp_${exp}ms.raw --stream-count=1
done

# Process HDR
./hdr_processor exp_2.5ms.raw exp_10ms.raw exp_40ms.raw hdr_output.raw

# Evaluate
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

# Load exposures
exp1 = np.fromfile('exp_2.5ms.raw', dtype=np.uint16).reshape((1080, 1920))
exp2 = np.fromfile('exp_10ms.raw', dtype=np.uint16).reshape((1080, 1920))
exp3 = np.fromfile('exp_40ms.raw', dtype=np.uint16).reshape((1080, 1920))
hdr = np.fromfile('hdr_output.raw', dtype=np.uint16).reshape((1080, 1920))

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].hist(exp1.ravel(), bins=256, range=(0, 1024))
axes[0,0].set_title('Short Exposure (2.5ms)')

axes[0,1].hist(exp2.ravel(), bins=256, range=(0, 1024))
axes[0,1].set_title('Medium Exposure (10ms)')

axes[1,0].hist(exp3.ravel(), bins=256, range=(0, 1024))
axes[1,0].set_title('Long Exposure (40ms)')

axes[1,1].hist(hdr.ravel(), bins=256, range=(0, 1024))
axes[1,1].set_title('HDR Merged')

plt.tight_layout()
plt.savefig('hdr_histograms.png')
plt.show()
EOF
```

### Lab 2: Tone Mapping Comparison

```python
#!/usr/bin/env python3
"""
compare_tonemapping.py
Compare different tone mapping operators
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_hdr(filename):
    """Load HDR image (OpenEXR or custom format)"""
    # Simplified - load as float32
    return np.fromfile(filename, dtype=np.float32)

def tonemap_reinhard(hdr, key=0.18):
    """Reinhard global operator"""
    log_avg = np.exp(np.mean(np.log(hdr + 1e-6)))
    scaled = hdr * (key / log_avg)
    return scaled / (1 + scaled)

def tonemap_aces(hdr):
    """ACES filmic"""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0, 1)

def tonemap_hable(hdr):
    """Uncharted 2 / Hable"""
    def hable_func(x):
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
    
    exposure_bias = 2.0
    curr = hable_func(hdr * exposure_bias)
    white_scale = 1.0 / hable_func(11.2)
    return curr * white_scale

# Load HDR
hdr = load_hdr('hdr_image.hdr').reshape((1080, 1920))

# Apply tone mapping
ldr_reinhard = tonemap_reinhard(hdr)
ldr_aces = tonemap_aces(hdr)
ldr_hable = tonemap_hable(hdr)

# Display comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].imshow(np.clip(hdr, 0, 1), cmap='gray')
axes[0,0].set_title('HDR (clipped for display)')

axes[0,1].imshow(ldr_reinhard, cmap='gray')
axes[0,1].set_title('Reinhard')

axes[1,0].imshow(ldr_aces, cmap='gray')
axes[1,0].set_title('ACES')

axes[1,1].imshow(ldr_hable, cmap='gray')
axes[1,1].set_title('Hable (Uncharted 2)')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.savefig('tonemap_comparison.png', dpi=150)
plt.show()
```

---

## 🐛 Debugging Techniques

### Debug 1: HDR Ghosting

**Symptoms:** Double images, motion trails in HDR output.

**Detection:**

```c
/**
 * @brief Detect motion between HDR exposures
 */
static float detect_hdr_motion(
    uint16_t *img1,
    uint16_t *img2,
    unsigned int width,
    unsigned int height)
{
    uint64_t diff_sum = 0;
    unsigned int pixels = width * height;
    
    for (unsigned int i = 0; i < pixels; i++) {
        int diff = abs((int)img1[i] - (int)img2[i]);
        diff_sum += diff;
    }
    
    float avg_diff = (float)diff_sum / pixels;
    
    return avg_diff;
}

/**
 * @brief Motion-compensated HDR merge
 */
static void hdr_merge_with_motion_detection(
    struct hdr_capture *hdr,
    float *output,
    float motion_threshold)
{
    /* Detect motion between exposures */
    for (unsigned int i = 1; i < hdr->num_images; i++) {
        float motion = detect_hdr_motion(
            hdr->images[0], hdr->images[i],
            hdr->width, hdr->height
        );
        
        if (motion > motion_threshold) {
            printf("Warning: Motion detected between exposures (%.1f)\n", motion);
            /* Use only first exposure or apply motion compensation */
        }
    }
    
    /* Proceed with merge */
    hdr_merge(hdr, output);
}
```

### Debug 2: WDR Artifacts

**Analysis Tool:**

```python
#!/usr/bin/env python3
"""
analyze_wdr_artifacts.py
Detect WDR stitching artifacts
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_stitching_artifacts(image, threshold_percentile=95):
    """Detect WDR stitching boundaries"""
    
    # Calculate gradient
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find high-gradient regions (potential artifacts)
    threshold = np.percentile(gradient, threshold_percentile)
    artifacts = gradient > threshold
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('WDR Image')
    
    axes[1].imshow(gradient, cmap='hot')
    axes[1].set_title('Gradient Magnitude')
    
    axes[2].imshow(artifacts, cmap='gray')
    axes[2].set_title('Potential Artifacts')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('wdr_artifacts.png')
    plt.show()
    
    artifact_percentage = np.sum(artifacts) / artifacts.size * 100
    print(f"Artifact pixels: {artifact_percentage:.2f}%")

# Load WDR image
wdr = cv2.imread('wdr_image.png', cv2.IMREAD_GRAYSCALE)
detect_stitching_artifacts(wdr)
```

---

## ⚡ Performance Optimization

### Optimization 1: GPU-Accelerated Tone Mapping

```cuda
/**
 * @file tonemap_cuda.cu
 * @brief CUDA-accelerated tone mapping
 */

__global__ void tonemap_aces_kernel(
    const float *hdr,
    uint16_t *ldr,
    unsigned int width,
    unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    unsigned int idx = y * width + x;
    float h = hdr[idx];
    
    /* ACES tone mapping */
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    float mapped = (h * (a * h + b)) / (h * (c * h + d) + e);
    mapped = fmaxf(0.0f, fminf(1.0f, mapped));
    
    ldr[idx] = (uint16_t)(mapped * 1023.0f);
}

extern "C" void tonemap_aces_cuda(
    float *h_hdr,
    uint16_t *h_ldr,
    unsigned int width,
    unsigned int height)
{
    size_t size_hdr = width * height * sizeof(float);
    size_t size_ldr = width * height * sizeof(uint16_t);
    
    float *d_hdr;
    uint16_t *d_ldr;
    
    cudaMalloc(&d_hdr, size_hdr);
    cudaMalloc(&d_ldr, size_ldr);
    
    cudaMemcpy(d_hdr, h_hdr, size_hdr, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    tonemap_aces_kernel<<<grid, block>>>(d_hdr, d_ldr, width, height);
    
    cudaMemcpy(h_ldr, d_ldr, size_ldr, cudaMemcpyDeviceToHost);
    
    cudaFree(d_hdr);
    cudaFree(d_ldr);
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the difference between HDR and WDR. When would you use each?**

2. **Why is tone mapping necessary for HDR images? What happens without it?**

3. **Calculate the dynamic range extension for:**
   - 3 exposures, 2 stops apart
   - Split-pixel WDR with 4x sensitivity ratio

4. **What causes ghosting in HDR images? How can it be prevented?**

5. **Compare global vs local tone mapping: advantages and disadvantages.**

### Practical Challenges

1. **Implement deghosting algorithm for HDR merge.**

2. **Design a real-time HDR system using sensor WDR mode.**

3. **Optimize tone mapping to run at 60fps for 4K.**

4. **Debug an HDR image with severe color shifts.**

---

## 📚 Further Reading & Resources

### Papers
- "Recovering High Dynamic Range Radiance Maps from Photographs" - Debevec & Malik
- "Exposure Fusion" - Mertens et al.
- "Photographic Tone Reproduction for Digital Images" - Reinhard et al.

### Standards
- [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100/) - HDR TV
- [SMPTE ST 2084](https://ieeexplore.ieee.org/document/7291452) - PQ curve

### Tools
- **Luminance HDR:** HDR creation and tone mapping
- **Photomatix:** Commercial HDR software
- **OpenCV:** cv::createTonemapReinhard, cv::createTonemapDurand

---

## 🎓 Summary

Today we covered:
- ✅ HDR vs WDR fundamentals and dynamic range concepts
- ✅ Multi-exposure HDR capture and fusion algorithms
- ✅ Tone mapping operators (global and local)
- ✅ Sensor WDR modes (split pixel, DCG, staggered)
- ✅ Complete HDR processing pipeline
- ✅ Real-time WDR implementation
- ✅ Ghosting detection and artifact debugging

**Key Takeaways:**
1. HDR extends dynamic range through multiple exposures or sensor techniques
2. Tone mapping is critical for displaying HDR on standard displays
3. WDR modes enable real-time operation but with limited DR extension
4. Motion handling is crucial for multi-exposure HDR quality

**Next:** Day 9 - Multi-Camera Systems and Synchronization

---

**Day 8 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
