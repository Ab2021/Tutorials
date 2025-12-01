# Day 13: Advanced AE/AWB Algorithms
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** advanced auto-exposure algorithms and metering modes
2. **Implement** sophisticated auto-white balance techniques
3. **Configure** scene detection and adaptive 3A
4. **Develop** flicker detection and anti-flicker algorithms
5. **Debug** exposure and color balance issues
6. **Optimize** 3A convergence speed and stability

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with exposure/gain control, AWB support
*   **Software:** Statistics collection, color science libraries
*   **Knowledge:** Photometry, colorimetry, control theory
*   **Tools:** Color charts (Macbeth), light meters

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Advanced Auto-Exposure

#### 1.1 Metering Modes

**Center-Weighted Metering:**

```c
/**
 * @brief Center-weighted metering
 */
static float ae_meter_center_weighted(
    uint8_t *image,
    unsigned int width,
    unsigned int height)
{
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    int center_x = width / 2;
    int center_y = height / 2;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            /* Calculate distance from center */
            float dx = (x - center_x) / (float)center_x;
            float dy = (y - center_y) / (float)center_y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            /* Gaussian weight: higher at center */
            float weight = expf(-dist * dist / 0.5f);
            
            uint8_t lum = image[y * width + x];
            weighted_sum += lum * weight;
            weight_sum += weight;
        }
    }
    
    return weighted_sum / weight_sum;
}
```

**Spot Metering:**

```c
/**
 * @brief Spot metering (small central area)
 */
static float ae_meter_spot(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int spot_size)
{
    unsigned int x_start = (width - spot_size) / 2;
    unsigned int y_start = (height - spot_size) / 2;
    
    uint64_t sum = 0;
    
    for (unsigned int y = y_start; y < y_start + spot_size; y++) {
        for (unsigned int x = x_start; x < x_start + spot_size; x++) {
            sum += image[y * width + x];
        }
    }
    
    return (float)sum / (spot_size * spot_size);
}
```

**Multi-Zone Evaluative Metering:**

```c
/**
 * @brief Multi-zone evaluative metering
 */
#define NUM_ZONES_X 8
#define NUM_ZONES_Y 6

struct zone_stats {
    float avg_luminance;
    float min_luminance;
    float max_luminance;
    float weight;
};

static float ae_meter_evaluative(
    uint8_t *image,
    unsigned int width,
    unsigned int height)
{
    struct zone_stats zones[NUM_ZONES_Y][NUM_ZONES_X];
    
    unsigned int zone_width = width / NUM_ZONES_X;
    unsigned int zone_height = height / NUM_ZONES_Y;
    
    /* Compute statistics for each zone */
    for (unsigned int zy = 0; zy < NUM_ZONES_Y; zy++) {
        for (unsigned int zx = 0; zx < NUM_ZONES_X; zx++) {
            unsigned int x_start = zx * zone_width;
            unsigned int y_start = zy * zone_height;
            
            uint64_t sum = 0;
            uint8_t min_val = 255, max_val = 0;
            
            for (unsigned int y = y_start; y < y_start + zone_height; y++) {
                for (unsigned int x = x_start; x < x_start + zone_width; x++) {
                    uint8_t val = image[y * width + x];
                    sum += val;
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
            }
            
            zones[zy][zx].avg_luminance = (float)sum / (zone_width * zone_height);
            zones[zy][zx].min_luminance = min_val;
            zones[zy][zx].max_luminance = max_val;
            
            /* Weight based on position and contrast */
            float center_dist = sqrtf(
                powf((zx - NUM_ZONES_X/2.0f) / (NUM_ZONES_X/2.0f), 2) +
                powf((zy - NUM_ZONES_Y/2.0f) / (NUM_ZONES_Y/2.0f), 2)
            );
            
            float contrast = max_val - min_val;
            
            zones[zy][zx].weight = (1.0f - center_dist * 0.5f) * 
                                  (1.0f + contrast / 255.0f);
        }
    }
    
    /* Weighted average */
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (unsigned int zy = 0; zy < NUM_ZONES_Y; zy++) {
        for (unsigned int zx = 0; zx < NUM_ZONES_X; zx++) {
            weighted_sum += zones[zy][zx].avg_luminance * zones[zy][zx].weight;
            weight_sum += zones[zy][zx].weight;
        }
    }
    
    return weighted_sum / weight_sum;
}
```

#### 1.2 Histogram-Based AE

**Histogram Analysis:**

```c
/**
 * @brief Compute luminance histogram
 */
static void compute_histogram(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    uint32_t *histogram)
{
    memset(histogram, 0, 256 * sizeof(uint32_t));
    
    for (unsigned int i = 0; i < width * height; i++) {
        histogram[image[i]]++;
    }
}

/**
 * @brief Histogram-based exposure adjustment
 */
static float ae_histogram_based(
    uint32_t *histogram,
    unsigned int total_pixels)
{
    /* Calculate cumulative histogram */
    uint32_t cumulative[256];
    cumulative[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cumulative[i] = cumulative[i-1] + histogram[i];
    }
    
    /* Find percentiles */
    uint32_t p1_threshold = total_pixels * 0.01f;  /* 1st percentile */
    uint32_t p99_threshold = total_pixels * 0.99f; /* 99th percentile */
    
    uint8_t p1 = 0, p99 = 255;
    
    for (int i = 0; i < 256; i++) {
        if (cumulative[i] >= p1_threshold && p1 == 0) {
            p1 = i;
        }
        if (cumulative[i] >= p99_threshold) {
            p99 = i;
            break;
        }
    }
    
    /* Check for clipping */
    float highlight_clip = (float)histogram[255] / total_pixels;
    float shadow_clip = (float)histogram[0] / total_pixels;
    
    /* Target: p99 at 90% of range, avoid clipping */
    float target_p99 = 230.0f;
    float current_p99 = p99;
    
    float adjustment = target_p99 / current_p99;
    
    /* Limit adjustment if clipping */
    if (highlight_clip > 0.01f) {
        adjustment = fminf(adjustment, 0.95f);  /* Reduce exposure */
    }
    if (shadow_clip > 0.05f) {
        adjustment = fmaxf(adjustment, 1.05f);  /* Increase exposure */
    }
    
    return adjustment;
}
```

#### 1.3 Exposure Compensation

**EV Compensation:**

```c
/**
 * @brief Apply exposure compensation
 */
struct ae_params {
    float target_luminance;  /* Target average (e.g., 128) */
    float ev_compensation;   /* EV offset (-2.0 to +2.0) */
    float min_exposure_us;
    float max_exposure_us;
    float min_gain;
    float max_gain;
};

static void ae_calculate_settings(
    float current_luminance,
    struct ae_params *params,
    float *exposure_us,
    float *gain)
{
    /* Calculate required adjustment */
    float target = params->target_luminance * powf(2.0f, params->ev_compensation);
    float ratio = target / current_luminance;
    
    /* Start with current settings */
    float new_exposure = *exposure_us * ratio;
    float new_gain = *gain;
    
    /* Clamp exposure */
    if (new_exposure > params->max_exposure_us) {
        /* Exposure maxed, increase gain */
        float overflow = new_exposure / params->max_exposure_us;
        new_exposure = params->max_exposure_us;
        new_gain *= overflow;
    } else if (new_exposure < params->min_exposure_us) {
        /* Exposure too low, decrease gain */
        float underflow = params->min_exposure_us / new_exposure;
        new_exposure = params->min_exposure_us;
        new_gain /= underflow;
    }
    
    /* Clamp gain */
    if (new_gain > params->max_gain) {
        new_gain = params->max_gain;
    } else if (new_gain < params->min_gain) {
        new_gain = params->min_gain;
    }
    
    *exposure_us = new_exposure;
    *gain = new_gain;
}
```

### 🔹 Part 2: Advanced Auto-White Balance

#### 2.1 Gray World Algorithm

**Enhanced Gray World:**

```c
/**
 * @brief Gray world AWB with outlier rejection
 */
struct awb_gains {
    float r_gain;
    float g_gain;
    float b_gain;
};

static void awb_gray_world(
    uint8_t *image_r,
    uint8_t *image_g,
    uint8_t *image_b,
    unsigned int width,
    unsigned int height,
    struct awb_gains *gains)
{
    uint64_t sum_r = 0, sum_g = 0, sum_b = 0;
    unsigned int count = 0;
    
    /* First pass: compute averages */
    for (unsigned int i = 0; i < width * height; i++) {
        sum_r += image_r[i];
        sum_g += image_g[i];
        sum_b += image_b[i];
        count++;
    }
    
    float avg_r = (float)sum_r / count;
    float avg_g = (float)sum_g / count;
    float avg_b = (float)sum_b / count;
    
    /* Second pass: reject outliers and recalculate */
    sum_r = sum_g = sum_b = 0;
    count = 0;
    
    float threshold = 50.0f;  /* Outlier threshold */
    
    for (unsigned int i = 0; i < width * height; i++) {
        float r = image_r[i];
        float g = image_g[i];
        float b = image_b[i];
        
        /* Reject saturated pixels */
        if (r > 250 || g > 250 || b > 250)
            continue;
        
        /* Reject very dark pixels */
        if (r < 10 && g < 10 && b < 10)
            continue;
        
        /* Reject pixels far from average */
        if (fabsf(r - avg_r) > threshold ||
            fabsf(g - avg_g) > threshold ||
            fabsf(b - avg_b) > threshold)
            continue;
        
        sum_r += r;
        sum_g += g;
        sum_b += b;
        count++;
    }
    
    if (count > 0) {
        avg_r = (float)sum_r / count;
        avg_g = (float)sum_g / count;
        avg_b = (float)sum_b / count;
    }
    
    /* Calculate gains */
    float avg = (avg_r + avg_g + avg_b) / 3.0f;
    
    gains->r_gain = avg / avg_r;
    gains->g_gain = avg / avg_g;
    gains->b_gain = avg / avg_b;
    
    /* Normalize to green */
    gains->r_gain /= gains->g_gain;
    gains->b_gain /= gains->g_gain;
    gains->g_gain = 1.0f;
}
```

#### 2.2 White Patch Algorithm

```c
/**
 * @brief White patch (max RGB) AWB
 */
static void awb_white_patch(
    uint8_t *image_r,
    uint8_t *image_g,
    uint8_t *image_b,
    unsigned int width,
    unsigned int height,
    struct awb_gains *gains)
{
    /* Find brightest pixels (top 1%) */
    uint32_t hist_r[256] = {0};
    uint32_t hist_g[256] = {0};
    uint32_t hist_b[256] = {0};
    
    for (unsigned int i = 0; i < width * height; i++) {
        hist_r[image_r[i]]++;
        hist_g[image_g[i]]++;
        hist_b[image_b[i]]++;
    }
    
    /* Find 99th percentile */
    unsigned int threshold = width * height * 0.99f;
    
    uint32_t cum_r = 0, cum_g = 0, cum_b = 0;
    uint8_t p99_r = 255, p99_g = 255, p99_b = 255;
    
    for (int i = 255; i >= 0; i--) {
        cum_r += hist_r[i];
        cum_g += hist_g[i];
        cum_b += hist_b[i];
        
        if (cum_r >= threshold && p99_r == 255) p99_r = i;
        if (cum_g >= threshold && p99_g == 255) p99_g = i;
        if (cum_b >= threshold && p99_b == 255) p99_b = i;
    }
    
    /* Calculate gains to make white patch neutral */
    float max_val = fmaxf(p99_r, fmaxf(p99_g, p99_b));
    
    gains->r_gain = max_val / p99_r;
    gains->g_gain = max_val / p99_g;
    gains->b_gain = max_val / p99_b;
    
    /* Normalize */
    gains->r_gain /= gains->g_gain;
    gains->b_gain /= gains->g_gain;
    gains->g_gain = 1.0f;
}
```

#### 2.3 Illuminant Estimation

**Gamut Mapping:**

```c
/**
 * @brief Gamut mapping AWB
 */

/* Canonical illuminants in RGB space */
static const struct {
    const char *name;
    float r, g, b;
} illuminants[] = {
    {"D65 (Daylight)", 1.00f, 1.00f, 1.00f},
    {"A (Tungsten)", 1.40f, 1.00f, 0.70f},
    {"F (Fluorescent)", 0.95f, 1.00f, 1.10f},
    {"Shade", 0.90f, 1.00f, 1.15f},
};

static void awb_gamut_mapping(
    uint8_t *image_r,
    uint8_t *image_g,
    uint8_t *image_b,
    unsigned int width,
    unsigned int height,
    struct awb_gains *gains)
{
    /* Build color distribution */
    float min_r = 255, max_r = 0;
    float min_g = 255, max_g = 0;
    float min_b = 255, max_b = 0;
    
    for (unsigned int i = 0; i < width * height; i++) {
        float r = image_r[i];
        float g = image_g[i];
        float b = image_b[i];
        
        /* Skip dark pixels */
        if (r + g + b < 30)
            continue;
        
        if (r < min_r) min_r = r;
        if (r > max_r) max_r = r;
        if (g < min_g) min_g = g;
        if (g > max_g) max_g = g;
        if (b < min_b) min_b = b;
        if (b > max_b) max_b = b;
    }
    
    /* Estimate illuminant by matching gamut */
    float best_match = FLT_MAX;
    int best_illuminant = 0;
    
    for (unsigned int i = 0; i < sizeof(illuminants)/sizeof(illuminants[0]); i++) {
        /* Calculate expected gamut under this illuminant */
        float exp_r_range = (max_r - min_r) / illuminants[i].r;
        float exp_g_range = (max_g - min_g) / illuminants[i].g;
        float exp_b_range = (max_b - min_b) / illuminants[i].b;
        
        /* Match score (lower is better) */
        float score = fabsf(exp_r_range - exp_g_range) +
                     fabsf(exp_g_range - exp_b_range) +
                     fabsf(exp_b_range - exp_r_range);
        
        if (score < best_match) {
            best_match = score;
            best_illuminant = i;
        }
    }
    
    /* Set gains based on estimated illuminant */
    gains->r_gain = 1.0f / illuminants[best_illuminant].r;
    gains->g_gain = 1.0f / illuminants[best_illuminant].g;
    gains->b_gain = 1.0f / illuminants[best_illuminant].b;
    
    /* Normalize */
    gains->r_gain /= gains->g_gain;
    gains->b_gain /= gains->g_gain;
    gains->g_gain = 1.0f;
}
```

### 🔹 Part 3: Flicker Detection and Mitigation

#### 3.1 Flicker Detection

```c
/**
 * @brief Detect AC flicker (50Hz/60Hz)
 */
struct flicker_detector {
    float luminance_history[60];  /* 2 seconds at 30fps */
    unsigned int history_idx;
    unsigned int history_count;
    
    bool flicker_detected;
    float flicker_frequency;
};

static void flicker_detect(
    struct flicker_detector *detector,
    float current_luminance)
{
    /* Add to history */
    detector->luminance_history[detector->history_idx] = current_luminance;
    detector->history_idx = (detector->history_idx + 1) % 60;
    
    if (detector->history_count < 60) {
        detector->history_count++;
        return;
    }
    
    /* FFT to detect periodic component */
    float *fft_input = malloc(60 * sizeof(float));
    memcpy(fft_input, detector->luminance_history, 60 * sizeof(float));
    
    /* Compute FFT (using library like FFTW) */
    float *fft_magnitude = compute_fft_1d(fft_input, 60);
    
    /* Look for peaks at 50Hz or 60Hz (assuming 30fps) */
    /* 50Hz flicker appears at 20Hz in 30fps video (50 - 30) */
    /* 60Hz flicker appears at 0Hz (invisible) or harmonics */
    
    float peak_50hz = fft_magnitude[20 * 60 / 30];  /* 20Hz bin */
    float peak_60hz = fft_magnitude[30 * 60 / 30];  /* 30Hz bin */
    
    float threshold = 0.1f;  /* Relative to DC component */
    
    if (peak_50hz > threshold) {
        detector->flicker_detected = true;
        detector->flicker_frequency = 50.0f;
    } else if (peak_60hz > threshold) {
        detector->flicker_detected = true;
        detector->flicker_frequency = 60.0f;
    } else {
        detector->flicker_detected = false;
    }
    
    free(fft_input);
    free(fft_magnitude);
}
```

#### 3.2 Anti-Flicker Exposure

```c
/**
 * @brief Calculate anti-flicker exposure time
 */
static float ae_anti_flicker_exposure(
    float desired_exposure_us,
    float flicker_frequency)
{
    /* Exposure should be multiple of flicker period */
    float flicker_period_us = 1000000.0f / flicker_frequency;
    
    /* Round to nearest multiple */
    int num_periods = (int)(desired_exposure_us / flicker_period_us + 0.5f);
    
    if (num_periods < 1)
        num_periods = 1;
    
    return num_periods * flicker_period_us;
}
```

---

## 💻 Implementation Examples

### Example 1: Complete 3A System

```c
/**
 * @file advanced_3a_system.c
 * @brief Advanced 3A (AE/AWB/AF) system
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

struct ae_state {
    enum {
        AE_SEARCHING,
        AE_CONVERGED,
        AE_LOCKED
    } state;
    
    float current_luminance;
    float target_luminance;
    
    float exposure_us;
    float gain;
    
    /* Metering */
    enum {
        METER_AVERAGE,
        METER_CENTER_WEIGHTED,
        METER_SPOT,
        METER_EVALUATIVE
    } metering_mode;
    
    /* Histogram */
    uint32_t histogram[256];
    
    /* Flicker */
    struct flicker_detector flicker;
};

struct awb_state {
    enum {
        AWB_SEARCHING,
        AWB_CONVERGED,
        AWB_LOCKED
    } state;
    
    struct awb_gains gains;
    
    /* Algorithm */
    enum {
        AWB_GRAY_WORLD,
        AWB_WHITE_PATCH,
        AWB_GAMUT_MAPPING
    } algorithm;
    
    /* Convergence */
    float prev_r_gain;
    float prev_b_gain;
    unsigned int stable_frames;
};

struct threea_system {
    struct ae_state ae;
    struct awb_state awb;
    
    /* Callbacks */
    int (*set_exposure)(float exposure_us);
    int (*set_gain)(float gain);
    int (*set_wb_gains)(float r_gain, float b_gain);
    
    /* Statistics */
    unsigned int frame_count;
};

/**
 * @brief Initialize 3A system
 */
static struct threea_system *threea_init(void)
{
    struct threea_system *sys = calloc(1, sizeof(*sys));
    
    /* Initialize AE */
    sys->ae.state = AE_SEARCHING;
    sys->ae.target_luminance = 128.0f;
    sys->ae.exposure_us = 10000.0f;  /* 10ms */
    sys->ae.gain = 1.0f;
    sys->ae.metering_mode = METER_EVALUATIVE;
    
    /* Initialize AWB */
    sys->awb.state = AWB_SEARCHING;
    sys->awb.gains.r_gain = 1.0f;
    sys->awb.gains.g_gain = 1.0f;
    sys->awb.gains.b_gain = 1.0f;
    sys->awb.algorithm = AWB_GRAY_WORLD;
    
    return sys;
}

/**
 * @brief Update AE
 */
static void threea_update_ae(
    struct threea_system *sys,
    uint8_t *image_y,
    unsigned int width,
    unsigned int height)
{
    /* Compute statistics */
    compute_histogram(image_y, width, height, sys->ae.histogram);
    
    /* Meter scene */
    switch (sys->ae.metering_mode) {
    case METER_AVERAGE:
        sys->ae.current_luminance = ae_meter_average(image_y, width, height);
        break;
    case METER_CENTER_WEIGHTED:
        sys->ae.current_luminance = ae_meter_center_weighted(image_y, width, height);
        break;
    case METER_SPOT:
        sys->ae.current_luminance = ae_meter_spot(image_y, width, height, 100);
        break;
    case METER_EVALUATIVE:
        sys->ae.current_luminance = ae_meter_evaluative(image_y, width, height);
        break;
    }
    
    /* Flicker detection */
    flicker_detect(&sys->ae.flicker, sys->ae.current_luminance);
    
    /* Calculate new settings */
    struct ae_params params = {
        .target_luminance = sys->ae.target_luminance,
        .ev_compensation = 0.0f,
        .min_exposure_us = 100.0f,
        .max_exposure_us = 33000.0f,  /* 1/30s */
        .min_gain = 1.0f,
        .max_gain = 16.0f
    };
    
    float new_exposure = sys->ae.exposure_us;
    float new_gain = sys->ae.gain;
    
    ae_calculate_settings(sys->ae.current_luminance, &params,
                         &new_exposure, &new_gain);
    
    /* Anti-flicker adjustment */
    if (sys->ae.flicker.flicker_detected) {
        new_exposure = ae_anti_flicker_exposure(new_exposure,
                                               sys->ae.flicker.flicker_frequency);
    }
    
    /* Smooth convergence */
    float alpha = 0.3f;  /* Convergence speed */
    sys->ae.exposure_us += (new_exposure - sys->ae.exposure_us) * alpha;
    sys->ae.gain += (new_gain - sys->ae.gain) * alpha;
    
    /* Apply settings */
    sys->set_exposure(sys->ae.exposure_us);
    sys->set_gain(sys->ae.gain);
    
    /* Check convergence */
    if (fabsf(sys->ae.current_luminance - sys->ae.target_luminance) < 5.0f) {
        sys->ae.state = AE_CONVERGED;
    } else {
        sys->ae.state = AE_SEARCHING;
    }
}

/**
 * @brief Update AWB
 */
static void threea_update_awb(
    struct threea_system *sys,
    uint8_t *image_r,
    uint8_t *image_g,
    uint8_t *image_b,
    unsigned int width,
    unsigned int height)
{
    struct awb_gains new_gains;
    
    /* Calculate gains */
    switch (sys->awb.algorithm) {
    case AWB_GRAY_WORLD:
        awb_gray_world(image_r, image_g, image_b, width, height, &new_gains);
        break;
    case AWB_WHITE_PATCH:
        awb_white_patch(image_r, image_g, image_b, width, height, &new_gains);
        break;
    case AWB_GAMUT_MAPPING:
        awb_gamut_mapping(image_r, image_g, image_b, width, height, &new_gains);
        break;
    }
    
    /* Smooth convergence */
    float alpha = 0.1f;  /* Slow convergence for stability */
    sys->awb.gains.r_gain += (new_gains.r_gain - sys->awb.gains.r_gain) * alpha;
    sys->awb.gains.b_gain += (new_gains.b_gain - sys->awb.gains.b_gain) * alpha;
    
    /* Apply gains */
    sys->set_wb_gains(sys->awb.gains.r_gain, sys->awb.gains.b_gain);
    
    /* Check convergence */
    float r_change = fabsf(sys->awb.gains.r_gain - sys->awb.prev_r_gain);
    float b_change = fabsf(sys->awb.gains.b_gain - sys->awb.prev_b_gain);
    
    if (r_change < 0.01f && b_change < 0.01f) {
        sys->awb.stable_frames++;
        if (sys->awb.stable_frames > 10) {
            sys->awb.state = AWB_CONVERGED;
        }
    } else {
        sys->awb.stable_frames = 0;
        sys->awb.state = AWB_SEARCHING;
    }
    
    sys->awb.prev_r_gain = sys->awb.gains.r_gain;
    sys->awb.prev_b_gain = sys->awb.gains.b_gain;
}

/**
 * @brief Update 3A system
 */
static void threea_update(
    struct threea_system *sys,
    uint8_t *image_y,
    uint8_t *image_r,
    uint8_t *image_g,
    uint8_t *image_b,
    unsigned int width,
    unsigned int height)
{
    /* Update AE */
    threea_update_ae(sys, image_y, width, height);
    
    /* Update AWB */
    threea_update_awb(sys, image_r, image_g, image_b, width, height);
    
    sys->frame_count++;
    
    /* Print status */
    if (sys->frame_count % 30 == 0) {
        printf("3A Status (frame %u):\n", sys->frame_count);
        printf("  AE: %s, Lum=%.1f, Exp=%.1fms, Gain=%.2f\n",
               sys->ae.state == AE_CONVERGED ? "CONVERGED" : "SEARCHING",
               sys->ae.current_luminance,
               sys->ae.exposure_us / 1000.0f,
               sys->ae.gain);
        printf("  AWB: %s, R=%.3f, B=%.3f\n",
               sys->awb.state == AWB_CONVERGED ? "CONVERGED" : "SEARCHING",
               sys->awb.gains.r_gain,
               sys->awb.gains.b_gain);
        if (sys->ae.flicker.flicker_detected) {
            printf("  Flicker: %.0fHz detected\n",
                   sys->ae.flicker.flicker_frequency);
        }
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Metering Mode Comparison

```python
#!/usr/bin/env python3
"""
compare_metering_modes.py
Compare different AE metering modes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def meter_average(image):
    return np.mean(image)

def meter_center_weighted(image):
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    
    # Gaussian weight centered
    center_y, center_x = h//2, w//2
    sigma = min(h, w) / 4
    
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    weight = np.exp(-(dist**2) / (2 * sigma**2))
    
    return np.sum(image * weight) / np.sum(weight)

def meter_spot(image, spot_size=100):
    h, w = image.shape
    y_start = (h - spot_size) // 2
    x_start = (w - spot_size) // 2
    
    spot = image[y_start:y_start+spot_size, x_start:x_start+spot_size]
    return np.mean(spot)

def meter_evaluative(image, zones_x=8, zones_y=6):
    h, w = image.shape
    zone_h = h // zones_y
    zone_w = w // zones_x
    
    weighted_sum = 0
    weight_sum = 0
    
    for zy in range(zones_y):
        for zx in range(zones_x):
            y_start = zy * zone_h
            x_start = zx * zone_w
            
            zone = image[y_start:y_start+zone_h, x_start:x_start+zone_w]
            zone_avg = np.mean(zone)
            zone_contrast = np.std(zone)
            
            # Weight by position and contrast
            center_dist = np.sqrt(
                ((zx - zones_x/2) / (zones_x/2))**2 +
                ((zy - zones_y/2) / (zones_y/2))**2
            )
            
            weight = (1 - center_dist * 0.5) * (1 + zone_contrast / 128)
            
            weighted_sum += zone_avg * weight
            weight_sum += weight
    
    return weighted_sum / weight_sum

# Test with backlit scene
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Compute metering
avg = meter_average(gray)
center = meter_center_weighted(gray)
spot = meter_spot(gray)
eval_meter = meter_evaluative(gray)

print("Metering Comparison:")
print(f"  Average: {avg:.1f}")
print(f"  Center-weighted: {center:.1f}")
print(f"  Spot: {spot:.1f}")
print(f"  Evaluative: {eval_meter:.1f}")

# Visualize weights
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title(f'Original (Avg={avg:.1f})')

# Center-weighted visualization
h, w = gray.shape
y, x = np.ogrid[:h, :w]
center_y, center_x = h//2, w//2
sigma = min(h, w) / 4
dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
weight = np.exp(-(dist**2) / (2 * sigma**2))

axes[0,1].imshow(weight, cmap='hot')
axes[0,1].set_title(f'Center-Weighted (Meter={center:.1f})')

# Spot visualization
spot_vis = np.zeros_like(gray)
spot_size = 100
y_start = (h - spot_size) // 2
x_start = (w - spot_size) // 2
spot_vis[y_start:y_start+spot_size, x_start:x_start+spot_size] = 255

axes[1,0].imshow(spot_vis, cmap='gray')
axes[1,0].set_title(f'Spot (Meter={spot:.1f})')

# Evaluative zones
axes[1,1].imshow(gray, cmap='gray')
zones_x, zones_y = 8, 6
zone_h, zone_w = h // zones_y, w // zones_x

for zy in range(zones_y + 1):
    axes[1,1].axhline(y=zy*zone_h, color='r', linewidth=0.5)
for zx in range(zones_x + 1):
    axes[1,1].axvline(x=zx*zone_w, color='r', linewidth=0.5)

axes[1,1].set_title(f'Evaluative (Meter={eval_meter:.1f})')

plt.tight_layout()
plt.savefig('metering_comparison.png')
plt.show()

cap.release()
```

### Lab 2: AWB Algorithm Comparison

```python
#!/usr/bin/env python3
"""
compare_awb_algorithms.py
Compare AWB algorithms
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def awb_gray_world(image):
    """Gray world AWB"""
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Reject saturated and dark pixels
    mask = (r < 250) & (g < 250) & (b < 250) & (r > 10) & (g > 10) & (b > 10)
    
    avg_r = np.mean(r[mask])
    avg_g = np.mean(g[mask])
    avg_b = np.mean(b[mask])
    
    avg = (avg_r + avg_g + avg_b) / 3
    
    r_gain = avg / avg_r
    g_gain = avg / avg_g
    b_gain = avg / avg_b
    
    # Normalize to green
    r_gain /= g_gain
    b_gain /= g_gain
    
    # Apply gains
    result = image.copy().astype(np.float32)
    result[:,:,2] *= r_gain
    result[:,:,0] *= b_gain
    
    return np.clip(result, 0, 255).astype(np.uint8)

def awb_white_patch(image):
    """White patch AWB"""
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Find 99th percentile
    p99_r = np.percentile(r, 99)
    p99_g = np.percentile(g, 99)
    p99_b = np.percentile(b, 99)
    
    max_val = max(p99_r, p99_g, p99_b)
    
    r_gain = max_val / p99_r
    g_gain = max_val / p99_g
    b_gain = max_val / p99_b
    
    # Apply gains
    result = image.copy().astype(np.float32)
    result[:,:,2] *= r_gain
    result[:,:,1] *= g_gain
    result[:,:,0] *= b_gain
    
    return np.clip(result, 0, 255).astype(np.uint8)

# Load test image
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Apply AWB algorithms
gray_world = awb_gray_world(frame)
white_patch = awb_white_patch(frame)

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(gray_world, cv2.COLOR_BGR2RGB))
axes[1].set_title('Gray World AWB')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(white_patch, cv2.COLOR_BGR2RGB))
axes[2].set_title('White Patch AWB')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('awb_comparison.png')
plt.show()
```

---

## 🐛 Debugging Techniques

### Debug 1: AE Oscillation

**Symptoms:** Exposure oscillates, never converges.

**Analysis:**

```c
/**
 * @brief Detect AE oscillation
 */
struct ae_oscillation_detector {
    float exposure_history[20];
    unsigned int idx;
    unsigned int count;
};

static bool detect_ae_oscillation(
    struct ae_oscillation_detector *detector,
    float current_exposure)
{
    detector->exposure_history[detector->idx] = current_exposure;
    detector->idx = (detector->idx + 1) % 20;
    
    if (detector->count < 20) {
        detector->count++;
        return false;
    }
    
    /* Count direction changes */
    unsigned int changes = 0;
    for (unsigned int i = 1; i < 19; i++) {
        float prev_diff = detector->exposure_history[i] - 
                         detector->exposure_history[i-1];
        float curr_diff = detector->exposure_history[i+1] - 
                         detector->exposure_history[i];
        
        if ((prev_diff > 0 && curr_diff < 0) ||
            (prev_diff < 0 && curr_diff > 0)) {
            changes++;
        }
    }
    
    /* Oscillating if > 5 changes in 20 samples */
    return changes > 5;
}
```

**Fix:** Reduce convergence speed (alpha), add deadband.

### Debug 2: AWB Color Cast

**Symptoms:** Persistent color cast in certain scenes.

**Debugging:**

```python
#!/usr/bin/env python3
"""
debug_awb_colorcast.py
Analyze AWB color cast issues
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_color_distribution(image):
    """Analyze color distribution"""
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Reject saturated and dark pixels
    mask = (r < 250) & (g < 250) & (b < 250) & (r > 10) & (g > 10) & (b > 10)
    
    r_valid = r[mask]
    g_valid = g[mask]
    b_valid = b[mask]
    
    print("Color Statistics:")
    print(f"  R: mean={np.mean(r_valid):.1f}, std={np.std(r_valid):.1f}")
    print(f"  G: mean={np.mean(g_valid):.1f}, std={np.std(g_valid):.1f}")
    print(f"  B: mean={np.mean(b_valid):.1f}, std={np.std(b_valid):.1f}")
    
    # Plot R/G and B/G ratios
    rg_ratio = r_valid / (g_valid + 1)
    bg_ratio = b_valid / (g_valid + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(rg_ratio, bg_ratio, alpha=0.1, s=1)
    plt.xlabel('R/G Ratio')
    plt.ylabel('B/G Ratio')
    plt.title('Color Distribution')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist2d(rg_ratio, bg_ratio, bins=50, cmap='hot')
    plt.xlabel('R/G Ratio')
    plt.ylabel('B/G Ratio')
    plt.title('Color Density')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('color_distribution.png')
    plt.show()

# Load image
image = cv2.imread('test_image.jpg')
analyze_color_distribution(image)
```

---

## ⚡ Performance Optimization

### Optimization 1: Fast Histogram Computation

```c
/**
 * @brief SIMD-optimized histogram computation
 */
#include <immintrin.h>  /* AVX2 */

static void compute_histogram_simd(
    uint8_t *image,
    unsigned int size,
    uint32_t *histogram)
{
    memset(histogram, 0, 256 * sizeof(uint32_t));
    
    /* Process 32 pixels at a time with AVX2 */
    unsigned int i;
    for (i = 0; i + 32 <= size; i += 32) {
        __m256i pixels = _mm256_loadu_si256((__m256i*)&image[i]);
        
        /* Extract bytes and increment histogram */
        uint8_t *p = (uint8_t*)&pixels;
        for (int j = 0; j < 32; j++) {
            histogram[p[j]]++;
        }
    }
    
    /* Process remaining pixels */
    for (; i < size; i++) {
        histogram[image[i]]++;
    }
}
```

### Optimization 2: Lookup Table for AWB

```c
/**
 * @brief Precomputed AWB gain LUT
 */
struct awb_lut {
    uint8_t r_table[256];
    uint8_t b_table[256];
};

static void awb_build_lut(
    struct awb_lut *lut,
    float r_gain,
    float b_gain)
{
    for (int i = 0; i < 256; i++) {
        int r_val = (int)(i * r_gain);
        int b_val = (int)(i * b_gain);
        
        lut->r_table[i] = (r_val > 255) ? 255 : r_val;
        lut->b_table[i] = (b_val > 255) ? 255 : b_val;
    }
}

static void awb_apply_lut(
    uint8_t *image_r,
    uint8_t *image_b,
    unsigned int size,
    struct awb_lut *lut)
{
    for (unsigned int i = 0; i < size; i++) {
        image_r[i] = lut->r_table[image_r[i]];
        image_b[i] = lut->b_table[image_b[i]];
    }
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the difference between center-weighted and evaluative metering.**

2. **Why does gray world AWB fail in scenes with dominant colors?**

3. **What causes AC flicker in camera images? How to detect and mitigate?**

4. **Compare histogram-based vs average-based AE.**

5. **Design a 3A system for:**
   - Low-light photography
   - High-speed sports
   - Studio portrait

### Practical Challenges

1. **Implement face-detection-based AE metering.**

2. **Debug AWB that produces blue cast indoors.**

3. **Optimize 3A to converge in < 10 frames.**

4. **Design anti-flicker algorithm for LED lighting.**

---

## 📚 Further Reading & Resources

### Papers
- "Auto-Exposure Control for Digital Cameras" - Various
- "Color Constancy Algorithms" - Finlayson et al.

### Books
- "Digital Camera Image Processing" - Nakamura
- "Color Science" - Wyszecki & Stiles

### Tools
- **libcamera:** 3A algorithms
- **OpenCV:** Color space conversions
- **ColorChecker:** AWB calibration

---

## 🎓 Summary

Today we covered:
- ✅ Advanced AE metering modes (center-weighted, spot, evaluative)
- ✅ Histogram-based exposure control
- ✅ Advanced AWB algorithms (gray world, white patch, gamut mapping)
- ✅ Flicker detection and anti-flicker exposure
- ✅ Complete 3A system integration
- ✅ Performance optimization techniques

**Key Takeaways:**
1. Metering mode significantly affects exposure in complex scenes
2. AWB algorithm choice depends on scene characteristics
3. Flicker detection is critical for indoor lighting
4. Smooth convergence prevents oscillation
5. Histogram analysis provides better exposure control

**Next:** Day 14 - Week 2 Review and Integration Project

---

**Day 13 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
