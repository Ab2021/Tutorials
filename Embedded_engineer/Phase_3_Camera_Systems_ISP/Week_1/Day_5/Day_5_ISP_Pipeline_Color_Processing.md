# Day 5: ISP Pipeline Architecture and Color Processing
## Phase 3: Camera Systems & ISP | Week 1: Camera Fundamentals

---

## 🎯 Learning Objectives
1. **Understand** complete ISP pipeline architecture and stages
2. **Implement** color correction matrix (CCM) and white balance
3. **Configure** gamma correction and tone mapping
4. **Develop** lens shading correction (LSC) algorithms
5. **Debug** ISP pipeline issues and artifacts
6. **Optimize** ISP performance and memory bandwidth

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Platform with ISP hardware or software ISP capability
*   **Software:** Linux kernel with V4L2 media controller, ISP drivers
*   **Knowledge:** Color science basics, matrix operations, image processing
*   **Tools:** Color checker chart, spectrophotometer (optional)

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: ISP Pipeline Overview

#### 1.1 Complete ISP Architecture

**Typical ISP Pipeline Stages:**

```mermaid
graph LR
    A[RAW Bayer] --> B[Black Level Correction]
    B --> C[Lens Shading Correction]
    C --> D[Bad Pixel Correction]
    D --> E[Noise Reduction]
    E --> F[Demosaic]
    F --> G[Color Correction Matrix]
    G --> H[Gamma Correction]
    H --> I[Color Space Conversion]
    I --> J[Sharpening]
    J --> K[RGB/YUV Output]
```

**Stage Categories:**

1. **RAW Domain Processing:**
   - Black level correction (BLC)
   - Lens shading correction (LSC)
   - Bad pixel correction (BPC)
   - Noise reduction (NR)
   - White balance (WB)

2. **RGB Domain Processing:**
   - Demosaicing
   - Color correction matrix (CCM)
   - Gamma correction
   - Tone mapping

3. **YUV Domain Processing:**
   - Color space conversion
   - Chroma noise reduction
   - Edge enhancement
   - Scaling/cropping

#### 1.2 Memory Architecture

**Buffer Flow:**

```
Sensor → DMA → RAW Buffer → ISP → RGB Buffer → Encoder/Display
         ↓                    ↓
    Statistics          Histogram/AWB/AE
```

**Memory Bandwidth Calculation:**

```c
/**
 * @brief Calculate ISP memory bandwidth requirements
 */
struct isp_bandwidth {
    unsigned long read_bw;   /* Bytes/second read */
    unsigned long write_bw;  /* Bytes/second write */
    unsigned long total_bw;
};

static struct isp_bandwidth calc_isp_bandwidth(
    unsigned int width,
    unsigned int height,
    unsigned int fps,
    unsigned int bpp_in,   /* Input bits per pixel */
    unsigned int bpp_out)  /* Output bits per pixel */
{
    struct isp_bandwidth bw = {0};
    unsigned long pixels_per_sec = width * height * fps;
    
    /* Input bandwidth (RAW read) */
    bw.read_bw = pixels_per_sec * bpp_in / 8;
    
    /* Output bandwidth (RGB/YUV write) */
    bw.write_bw = pixels_per_sec * bpp_out / 8;
    
    /* Statistics read (typically 1-2% of image data) */
    bw.read_bw += pixels_per_sec * 2 / 100;
    
    /* Total */
    bw.total_bw = bw.read_bw + bw.write_bw;
    
    return bw;
}

/* Example: 1080p@30fps, RAW10 → RGB24 */
/* Read: 1920×1080×30×10/8 = 77.76 MB/s */
/* Write: 1920×1080×30×24/8 = 186.62 MB/s */
/* Total: ~265 MB/s */
```

### 🔹 Part 2: Color Processing Fundamentals

#### 2.1 Black Level Correction (BLC)

**Purpose:** Remove sensor dark current offset.

**Algorithm:**

```c
/**
 * @brief Black level correction
 * @param raw Input RAW image
 * @param black_level Black level value (typically 64 for 10-bit)
 */
static void isp_black_level_correction(
    uint16_t *raw,
    unsigned int width,
    unsigned int height,
    uint16_t black_level)
{
    unsigned int total_pixels = width * height;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        if (raw[i] > black_level) {
            raw[i] -= black_level;
        } else {
            raw[i] = 0;
        }
    }
}

/* Per-channel black level for Bayer pattern */
struct blc_params {
    uint16_t r;   /* Red channel */
    uint16_t gr;  /* Green-Red channel */
    uint16_t gb;  /* Green-Blue channel */
    uint16_t b;   /* Blue channel */
};

static void isp_blc_bayer(
    uint16_t *raw,
    unsigned int width,
    unsigned int height,
    struct blc_params *blc)
{
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            uint16_t bl;
            
            /* Determine channel based on Bayer pattern (RGGB) */
            if ((y % 2) == 0) {
                bl = (x % 2) == 0 ? blc->r : blc->gr;
            } else {
                bl = (x % 2) == 0 ? blc->gb : blc->b;
            }
            
            if (raw[idx] > bl) {
                raw[idx] -= bl;
            } else {
                raw[idx] = 0;
            }
        }
    }
}
```

#### 2.2 Lens Shading Correction (LSC)

**Problem:** Vignetting causes brightness falloff toward image corners.

**Radial LSC Model:**

```c
/**
 * @brief Radial lens shading correction
 */
struct lsc_radial_params {
    float center_x;  /* Optical center X (normalized 0-1) */
    float center_y;  /* Optical center Y (normalized 0-1) */
    float k1, k2, k3;  /* Radial coefficients */
};

static float lsc_radial_gain(
    float x, float y,  /* Normalized coordinates */
    struct lsc_radial_params *params)
{
    float dx = x - params->center_x;
    float dy = y - params->center_y;
    float r2 = dx*dx + dy*dy;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    /* Gain = 1 + k1*r^2 + k2*r^4 + k3*r^6 */
    float gain = 1.0f + params->k1 * r2 + 
                        params->k2 * r4 + 
                        params->k3 * r6;
    
    return gain;
}

static void isp_lsc_radial(
    uint16_t *raw,
    unsigned int width,
    unsigned int height,
    struct lsc_radial_params *params)
{
    for (unsigned int y = 0; y < height; y++) {
        float norm_y = (float)y / height;
        
        for (unsigned int x = 0; x < width; x++) {
            float norm_x = (float)x / width;
            float gain = lsc_radial_gain(norm_x, norm_y, params);
            
            unsigned int idx = y * width + x;
            uint16_t corrected = (uint16_t)(raw[idx] * gain);
            
            /* Clamp to max value */
            raw[idx] = (corrected > 1023) ? 1023 : corrected;
        }
    }
}
```

**Grid-based LSC:**

```c
/**
 * @brief Grid-based lens shading correction (more accurate)
 */
#define LSC_GRID_WIDTH  17
#define LSC_GRID_HEIGHT 13

struct lsc_grid_params {
    float r_gain[LSC_GRID_HEIGHT][LSC_GRID_WIDTH];
    float gr_gain[LSC_GRID_HEIGHT][LSC_GRID_WIDTH];
    float gb_gain[LSC_GRID_HEIGHT][LSC_GRID_WIDTH];
    float b_gain[LSC_GRID_HEIGHT][LSC_GRID_WIDTH];
};

static float bilinear_interpolate(
    float q11, float q12, float q21, float q22,
    float x, float y)
{
    float r1 = q11 * (1 - x) + q21 * x;
    float r2 = q12 * (1 - x) + q22 * x;
    return r1 * (1 - y) + r2 * y;
}

static void isp_lsc_grid(
    uint16_t *raw,
    unsigned int width,
    unsigned int height,
    struct lsc_grid_params *lsc)
{
    float grid_step_x = (float)width / (LSC_GRID_WIDTH - 1);
    float grid_step_y = (float)height / (LSC_GRID_HEIGHT - 1);
    
    for (unsigned int y = 0; y < height; y++) {
        unsigned int grid_y = y / grid_step_y;
        float frac_y = (y - grid_y * grid_step_y) / grid_step_y;
        
        for (unsigned int x = 0; x < width; x++) {
            unsigned int grid_x = x / grid_step_x;
            float frac_x = (x - grid_x * grid_step_x) / grid_step_x;
            
            /* Get channel-specific gain grid */
            float (*gain_grid)[LSC_GRID_WIDTH];
            if ((y % 2) == 0) {
                gain_grid = (x % 2) == 0 ? lsc->r_gain : lsc->gr_gain;
            } else {
                gain_grid = (x % 2) == 0 ? lsc->gb_gain : lsc->b_gain;
            }
            
            /* Bilinear interpolation */
            float gain = bilinear_interpolate(
                gain_grid[grid_y][grid_x],
                gain_grid[grid_y+1][grid_x],
                gain_grid[grid_y][grid_x+1],
                gain_grid[grid_y+1][grid_x+1],
                frac_x, frac_y
            );
            
            unsigned int idx = y * width + x;
            uint16_t corrected = (uint16_t)(raw[idx] * gain);
            raw[idx] = (corrected > 1023) ? 1023 : corrected;
        }
    }
}
```

#### 2.3 White Balance

**Gray World Algorithm:**

```c
/**
 * @brief Auto white balance using gray world assumption
 */
struct wb_gains {
    float r_gain;
    float g_gain;
    float b_gain;
};

static struct wb_gains calculate_wb_gray_world(
    uint16_t *raw,
    unsigned int width,
    unsigned int height)
{
    uint64_t r_sum = 0, g_sum = 0, b_sum = 0;
    uint32_t r_count = 0, g_count = 0, b_count = 0;
    
    /* Accumulate channel sums */
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            
            if ((y % 2) == 0) {
                if ((x % 2) == 0) {
                    r_sum += raw[idx];
                    r_count++;
                } else {
                    g_sum += raw[idx];
                    g_count++;
                }
            } else {
                if ((x % 2) == 0) {
                    g_sum += raw[idx];
                    g_count++;
                } else {
                    b_sum += raw[idx];
                    b_count++;
                }
            }
        }
    }
    
    /* Calculate averages */
    float r_avg = (float)r_sum / r_count;
    float g_avg = (float)g_sum / g_count;
    float b_avg = (float)b_sum / b_count;
    
    /* Calculate gains (normalize to green) */
    struct wb_gains gains;
    gains.r_gain = g_avg / r_avg;
    gains.g_gain = 1.0f;
    gains.b_gain = g_avg / b_avg;
    
    return gains;
}

/**
 * @brief Apply white balance gains
 */
static void apply_wb_gains(
    uint16_t *raw,
    unsigned int width,
    unsigned int height,
    struct wb_gains *gains)
{
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            float gain;
            
            /* Select gain based on Bayer position */
            if ((y % 2) == 0) {
                gain = (x % 2) == 0 ? gains->r_gain : gains->g_gain;
            } else {
                gain = (x % 2) == 0 ? gains->g_gain : gains->b_gain;
            }
            
            uint16_t corrected = (uint16_t)(raw[idx] * gain);
            raw[idx] = (corrected > 1023) ? 1023 : corrected;
        }
    }
}
```

#### 2.4 Color Correction Matrix (CCM)

**Purpose:** Convert sensor RGB to standard color space (sRGB, Adobe RGB).

**Matrix Operation:**

```
[R']   [M00 M01 M02]   [R]
[G'] = [M10 M11 M12] × [G]
[B']   [M20 M21 M22]   [B]
```

**Implementation:**

```c
/**
 * @brief 3x3 Color Correction Matrix
 */
struct ccm_matrix {
    float m[3][3];
};

/* Example CCM for daylight illumination */
static const struct ccm_matrix ccm_d65 = {
    .m = {
        { 1.5234, -0.3789, -0.1445},
        {-0.2891,  1.4102, -0.1211},
        {-0.0391, -0.4297,  1.4688}
    }
};

/**
 * @brief Apply CCM to RGB pixel
 */
static void apply_ccm_pixel(
    uint16_t *r, uint16_t *g, uint16_t *b,
    const struct ccm_matrix *ccm,
    unsigned int max_val)
{
    float r_in = (float)*r;
    float g_in = (float)*g;
    float b_in = (float)*b;
    
    float r_out = ccm->m[0][0] * r_in + 
                  ccm->m[0][1] * g_in + 
                  ccm->m[0][2] * b_in;
    
    float g_out = ccm->m[1][0] * r_in + 
                  ccm->m[1][1] * g_in + 
                  ccm->m[1][2] * b_in;
    
    float b_out = ccm->m[2][0] * r_in + 
                  ccm->m[2][1] * g_in + 
                  ccm->m[2][2] * b_in;
    
    /* Clamp to valid range */
    *r = (r_out < 0) ? 0 : ((r_out > max_val) ? max_val : (uint16_t)r_out);
    *g = (g_out < 0) ? 0 : ((g_out > max_val) ? max_val : (uint16_t)g_out);
    *b = (b_out < 0) ? 0 : ((b_out > max_val) ? max_val : (uint16_t)b_out);
}

/**
 * @brief Apply CCM to entire RGB image
 */
static void isp_apply_ccm(
    uint16_t *r_plane,
    uint16_t *g_plane,
    uint16_t *b_plane,
    unsigned int width,
    unsigned int height,
    const struct ccm_matrix *ccm)
{
    unsigned int total_pixels = width * height;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        apply_ccm_pixel(&r_plane[i], &g_plane[i], &b_plane[i],
                       ccm, 1023);
    }
}
```

**CCM Calibration:**

```c
/**
 * @brief Calculate CCM from color checker measurements
 * 
 * Uses least-squares method to find CCM that best maps
 * measured colors to reference colors.
 */
struct color_sample {
    float r, g, b;          /* Measured RGB */
    float r_ref, g_ref, b_ref;  /* Reference RGB */
};

static struct ccm_matrix calibrate_ccm(
    struct color_sample *samples,
    unsigned int num_samples)
{
    /* This would use least-squares matrix solving */
    /* Simplified example - in practice use LAPACK or similar */
    
    struct ccm_matrix ccm;
    
    /* Build normal equations: A^T × A × x = A^T × b */
    /* Where A is measured colors, b is reference colors */
    /* Solve for x (CCM coefficients) */
    
    /* ... matrix math ... */
    
    return ccm;
}
```

#### 2.5 Gamma Correction

**Purpose:** Convert linear RGB to perceptual (gamma-encoded) RGB.

**sRGB Gamma:**

```c
/**
 * @brief sRGB gamma encoding
 */
static float srgb_gamma_encode(float linear)
{
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * powf(linear, 1.0f/2.4f) - 0.055f;
    }
}

/**
 * @brief Build gamma lookup table (LUT)
 */
#define GAMMA_LUT_SIZE 1024

static void build_gamma_lut(uint16_t *lut, float gamma)
{
    for (int i = 0; i < GAMMA_LUT_SIZE; i++) {
        float normalized = (float)i / (GAMMA_LUT_SIZE - 1);
        float gamma_corrected = powf(normalized, 1.0f / gamma);
        lut[i] = (uint16_t)(gamma_corrected * (GAMMA_LUT_SIZE - 1));
    }
}

/**
 * @brief Apply gamma correction using LUT
 */
static void isp_apply_gamma(
    uint16_t *rgb,
    unsigned int width,
    unsigned int height,
    uint16_t *gamma_lut,
    unsigned int input_max)
{
    unsigned int total_pixels = width * height * 3;  /* RGB */
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        /* Scale to LUT range */
        unsigned int lut_idx = (rgb[i] * (GAMMA_LUT_SIZE - 1)) / input_max;
        rgb[i] = gamma_lut[lut_idx];
    }
}
```

**Tone Mapping (HDR):**

```c
/**
 * @brief Simple Reinhard tone mapping
 */
static void tone_map_reinhard(
    uint16_t *rgb,
    unsigned int width,
    unsigned int height,
    float max_luminance)
{
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = (y * width + x) * 3;
            
            /* Calculate luminance */
            float r = rgb[idx + 0] / 1023.0f;
            float g = rgb[idx + 1] / 1023.0f;
            float b = rgb[idx + 2] / 1023.0f;
            
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            
            /* Reinhard operator */
            float lum_mapped = lum / (1.0f + lum / max_luminance);
            
            /* Scale RGB by luminance ratio */
            float scale = lum_mapped / (lum + 1e-6f);
            
            rgb[idx + 0] = (uint16_t)(r * scale * 1023);
            rgb[idx + 1] = (uint16_t)(g * scale * 1023);
            rgb[idx + 2] = (uint16_t)(b * scale * 1023);
        }
    }
}
```

### 🔹 Part 3: Color Space Conversion

#### 3.1 RGB to YUV Conversion

**BT.601 Standard:**

```
Y  =  0.299R + 0.587G + 0.114B
Cb = -0.169R - 0.331G + 0.500B + 128
Cr =  0.500R - 0.419G - 0.081B + 128
```

**Fixed-Point Implementation:**

```c
/**
 * @brief RGB to YUV conversion (BT.601)
 * Using fixed-point arithmetic for efficiency
 */
#define FP_SHIFT 10
#define FP_ONE (1 << FP_SHIFT)

/* Fixed-point coefficients */
#define FP_0_299  306   /* 0.299 * 1024 */
#define FP_0_587  601   /* 0.587 * 1024 */
#define FP_0_114  117   /* 0.114 * 1024 */
#define FP_0_169  173   /* 0.169 * 1024 */
#define FP_0_331  339   /* 0.331 * 1024 */
#define FP_0_500  512   /* 0.500 * 1024 */
#define FP_0_419  429   /* 0.419 * 1024 */
#define FP_0_081  83    /* 0.081 * 1024 */

static void rgb_to_yuv_bt601(
    uint8_t r, uint8_t g, uint8_t b,
    uint8_t *y, uint8_t *u, uint8_t *v)
{
    int32_t y_val = (FP_0_299 * r + FP_0_587 * g + FP_0_114 * b) >> FP_SHIFT;
    int32_t u_val = ((-FP_0_169 * r - FP_0_331 * g + FP_0_500 * b) >> FP_SHIFT) + 128;
    int32_t v_val = ((FP_0_500 * r - FP_0_419 * g - FP_0_081 * b) >> FP_SHIFT) + 128;
    
    *y = (y_val < 0) ? 0 : ((y_val > 255) ? 255 : y_val);
    *u = (u_val < 0) ? 0 : ((u_val > 255) ? 255 : u_val);
    *v = (v_val < 0) ? 0 : ((v_val > 255) ? 255 : v_val);
}

/**
 * @brief Convert RGB image to YUV420 (NV12 format)
 */
static void isp_rgb_to_nv12(
    uint8_t *rgb,
    uint8_t *y_plane,
    uint8_t *uv_plane,
    unsigned int width,
    unsigned int height)
{
    /* Y plane (full resolution) */
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            unsigned int rgb_idx = (i * width + j) * 3;
            unsigned int y_idx = i * width + j;
            
            uint8_t r = rgb[rgb_idx + 0];
            uint8_t g = rgb[rgb_idx + 1];
            uint8_t b = rgb[rgb_idx + 2];
            
            uint8_t y, u, v;
            rgb_to_yuv_bt601(r, g, b, &y, &u, &v);
            
            y_plane[y_idx] = y;
        }
    }
    
    /* UV plane (half resolution, interleaved) */
    for (unsigned int i = 0; i < height; i += 2) {
        for (unsigned int j = 0; j < width; j += 2) {
            /* Average 2x2 block */
            uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
            
            for (unsigned int di = 0; di < 2; di++) {
                for (unsigned int dj = 0; dj < 2; dj++) {
                    unsigned int rgb_idx = ((i+di) * width + (j+dj)) * 3;
                    r_sum += rgb[rgb_idx + 0];
                    g_sum += rgb[rgb_idx + 1];
                    b_sum += rgb[rgb_idx + 2];
                }
            }
            
            uint8_t r_avg = r_sum / 4;
            uint8_t g_avg = g_sum / 4;
            uint8_t b_avg = b_sum / 4;
            
            uint8_t y, u, v;
            rgb_to_yuv_bt601(r_avg, g_avg, b_avg, &y, &u, &v);
            
            unsigned int uv_idx = (i / 2) * width + j;
            uv_plane[uv_idx + 0] = u;
            uv_plane[uv_idx + 1] = v;
        }
    }
}
```

---

## 💻 Implementation Examples

### Example 1: Software ISP Pipeline

```c
/**
 * @file software_isp.c
 * @brief Complete software ISP implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

struct isp_config {
    /* Black level */
    struct blc_params blc;
    
    /* Lens shading */
    struct lsc_grid_params lsc;
    
    /* White balance */
    struct wb_gains wb;
    
    /* Color correction */
    struct ccm_matrix ccm;
    
    /* Gamma */
    uint16_t gamma_lut[GAMMA_LUT_SIZE];
};

struct isp_context {
    struct isp_config config;
    
    /* Buffers */
    uint16_t *raw_buffer;
    uint16_t *rgb_buffer;
    uint8_t *yuv_buffer;
    
    /* Dimensions */
    unsigned int width;
    unsigned int height;
};

/**
 * @brief Initialize ISP context
 */
static struct isp_context *isp_init(
    unsigned int width,
    unsigned int height)
{
    struct isp_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    
    ctx->width = width;
    ctx->height = height;
    
    /* Allocate buffers */
    ctx->raw_buffer = malloc(width * height * sizeof(uint16_t));
    ctx->rgb_buffer = malloc(width * height * 3 * sizeof(uint16_t));
    ctx->yuv_buffer = malloc(width * height * 3 / 2);  /* YUV420 */
    
    if (!ctx->raw_buffer || !ctx->rgb_buffer || !ctx->yuv_buffer) {
        free(ctx->raw_buffer);
        free(ctx->rgb_buffer);
        free(ctx->yuv_buffer);
        free(ctx);
        return NULL;
    }
    
    /* Initialize default configuration */
    /* BLC */
    ctx->config.blc.r = 64;
    ctx->config.blc.gr = 64;
    ctx->config.blc.gb = 64;
    ctx->config.blc.b = 64;
    
    /* WB (neutral) */
    ctx->config.wb.r_gain = 1.0f;
    ctx->config.wb.g_gain = 1.0f;
    ctx->config.wb.b_gain = 1.0f;
    
    /* CCM (identity) */
    memset(&ctx->config.ccm, 0, sizeof(ctx->config.ccm));
    ctx->config.ccm.m[0][0] = 1.0f;
    ctx->config.ccm.m[1][1] = 1.0f;
    ctx->config.ccm.m[2][2] = 1.0f;
    
    /* Gamma (2.2) */
    build_gamma_lut(ctx->config.gamma_lut, 2.2f);
    
    return ctx;
}

/**
 * @brief Process RAW image through ISP pipeline
 */
static int isp_process(struct isp_context *ctx)
{
    printf("ISP Processing: %ux%u\n", ctx->width, ctx->height);
    
    /* 1. Black level correction */
    printf("  [1/7] Black level correction...\n");
    isp_blc_bayer(ctx->raw_buffer, ctx->width, ctx->height,
                  &ctx->config.blc);
    
    /* 2. Lens shading correction */
    printf("  [2/7] Lens shading correction...\n");
    isp_lsc_grid(ctx->raw_buffer, ctx->width, ctx->height,
                 &ctx->config.lsc);
    
    /* 3. White balance */
    printf("  [3/7] White balance...\n");
    apply_wb_gains(ctx->raw_buffer, ctx->width, ctx->height,
                   &ctx->config.wb);
    
    /* 4. Demosaic */
    printf("  [4/7] Demosaicing...\n");
    /* Use demosaic from Day 2 */
    demosaic_bilinear(ctx->raw_buffer, ctx->rgb_buffer,
                     ctx->width, ctx->height);
    
    /* 5. Color correction matrix */
    printf("  [5/7] Color correction...\n");
    uint16_t *r_plane = ctx->rgb_buffer;
    uint16_t *g_plane = ctx->rgb_buffer + ctx->width * ctx->height;
    uint16_t *b_plane = ctx->rgb_buffer + ctx->width * ctx->height * 2;
    isp_apply_ccm(r_plane, g_plane, b_plane,
                  ctx->width, ctx->height, &ctx->config.ccm);
    
    /* 6. Gamma correction */
    printf("  [6/7] Gamma correction...\n");
    isp_apply_gamma(ctx->rgb_buffer, ctx->width, ctx->height,
                   ctx->config.gamma_lut, 1023);
    
    /* 7. Color space conversion */
    printf("  [7/7] RGB to YUV conversion...\n");
    /* Convert to 8-bit RGB first */
    uint8_t *rgb8 = malloc(ctx->width * ctx->height * 3);
    for (unsigned int i = 0; i < ctx->width * ctx->height * 3; i++) {
        rgb8[i] = ctx->rgb_buffer[i] >> 2;  /* 10-bit to 8-bit */
    }
    
    isp_rgb_to_nv12(rgb8, ctx->yuv_buffer,
                   ctx->yuv_buffer + ctx->width * ctx->height,
                   ctx->width, ctx->height);
    free(rgb8);
    
    printf("ISP Processing complete\n");
    return 0;
}

/**
 * @brief Example usage
 */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.raw> <output.yuv>\n", argv[0]);
        return 1;
    }
    
    /* Initialize ISP for 1080p */
    struct isp_context *ctx = isp_init(1920, 1080);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ISP\n");
        return 1;
    }
    
    /* Load RAW image */
    FILE *f_in = fopen(argv[1], "rb");
    if (!f_in) {
        perror("fopen input");
        return 1;
    }
    
    size_t read_size = fread(ctx->raw_buffer, sizeof(uint16_t),
                            ctx->width * ctx->height, f_in);
    fclose(f_in);
    
    if (read_size != ctx->width * ctx->height) {
        fprintf(stderr, "Failed to read RAW image\n");
        return 1;
    }
    
    /* Process */
    isp_process(ctx);
    
    /* Save YUV output */
    FILE *f_out = fopen(argv[2], "wb");
    if (!f_out) {
        perror("fopen output");
        return 1;
    }
    
    fwrite(ctx->yuv_buffer, 1, ctx->width * ctx->height * 3 / 2, f_out);
    fclose(f_out);
    
    printf("Output saved to %s\n", argv[2]);
    
    /* Cleanup */
    free(ctx->raw_buffer);
    free(ctx->rgb_buffer);
    free(ctx->yuv_buffer);
    free(ctx);
    
    return 0;
}
```

### Example 2: V4L2 ISP Driver Integration

```c
/**
 * @file v4l2_isp_driver.c
 * @brief V4L2 ISP driver with media controller
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <media/v4l2-device.h>
#include <media/v4l2-subdev.h>
#include <media/v4l2-ctrls.h>
#include <media/videobuf2-dma-contig.h>

#define ISP_NAME "custom-isp"

enum isp_pads {
    ISP_PAD_SINK,
    ISP_PAD_SOURCE,
    ISP_PAD_MAX
};

struct isp_device {
    struct v4l2_subdev subdev;
    struct media_pad pads[ISP_PAD_MAX];
    struct v4l2_ctrl_handler ctrl_handler;
    
    /* Hardware resources */
    void __iomem *base;
    struct clk *clk;
    
    /* Controls */
    struct v4l2_ctrl *brightness;
    struct v4l2_ctrl *contrast;
    struct v4l2_ctrl *saturation;
    
    /* State */
    bool streaming;
};

/**
 * @brief V4L2 control operations
 */
static int isp_s_ctrl(struct v4l2_ctrl *ctrl)
{
    struct isp_device *isp = container_of(ctrl->handler,
                                          struct isp_device,
                                          ctrl_handler);
    
    switch (ctrl->id) {
    case V4L2_CID_BRIGHTNESS:
        /* Program brightness register */
        writel(ctrl->val, isp->base + ISP_REG_BRIGHTNESS);
        break;
    
    case V4L2_CID_CONTRAST:
        writel(ctrl->val, isp->base + ISP_REG_CONTRAST);
        break;
    
    case V4L2_CID_SATURATION:
        writel(ctrl->val, isp->base + ISP_REG_SATURATION);
        break;
    
    default:
        return -EINVAL;
    }
    
    return 0;
}

static const struct v4l2_ctrl_ops isp_ctrl_ops = {
    .s_ctrl = isp_s_ctrl,
};

/**
 * @brief Initialize controls
 */
static int isp_init_controls(struct isp_device *isp)
{
    struct v4l2_ctrl_handler *hdl = &isp->ctrl_handler;
    
    v4l2_ctrl_handler_init(hdl, 3);
    
    isp->brightness = v4l2_ctrl_new_std(hdl, &isp_ctrl_ops,
                                        V4L2_CID_BRIGHTNESS,
                                        -128, 127, 1, 0);
    
    isp->contrast = v4l2_ctrl_new_std(hdl, &isp_ctrl_ops,
                                      V4L2_CID_CONTRAST,
                                      0, 255, 1, 128);
    
    isp->saturation = v4l2_ctrl_new_std(hdl, &isp_ctrl_ops,
                                        V4L2_CID_SATURATION,
                                        0, 255, 1, 128);
    
    if (hdl->error)
        return hdl->error;
    
    isp->subdev.ctrl_handler = hdl;
    
    return 0;
}

/**
 * @brief V4L2 subdev s_stream
 */
static int isp_s_stream(struct v4l2_subdev *sd, int enable)
{
    struct isp_device *isp = container_of(sd, struct isp_device, subdev);
    
    if (enable) {
        /* Enable ISP */
        writel(ISP_CTRL_ENABLE, isp->base + ISP_REG_CTRL);
        isp->streaming = true;
    } else {
        /* Disable ISP */
        writel(0, isp->base + ISP_REG_CTRL);
        isp->streaming = false;
    }
    
    return 0;
}

static const struct v4l2_subdev_video_ops isp_video_ops = {
    .s_stream = isp_s_stream,
};

static const struct v4l2_subdev_ops isp_subdev_ops = {
    .video = &isp_video_ops,
};

/**
 * @brief Platform driver probe
 */
static int isp_probe(struct platform_device *pdev)
{
    struct isp_device *isp;
    struct resource *res;
    int ret;
    
    isp = devm_kzalloc(&pdev->dev, sizeof(*isp), GFP_KERNEL);
    if (!isp)
        return -ENOMEM;
    
    /* Get resources */
    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    isp->base = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(isp->base))
        return PTR_ERR(isp->base);
    
    isp->clk = devm_clk_get(&pdev->dev, NULL);
    if (IS_ERR(isp->clk))
        return PTR_ERR(isp->clk);
    
    clk_prepare_enable(isp->clk);
    
    /* Initialize V4L2 subdev */
    v4l2_subdev_init(&isp->subdev, &isp_subdev_ops);
    isp->subdev.dev = &pdev->dev;
    snprintf(isp->subdev.name, sizeof(isp->subdev.name), ISP_NAME);
    
    /* Initialize controls */
    ret = isp_init_controls(isp);
    if (ret)
        goto err_clk;
    
    /* Initialize media entity */
    isp->pads[ISP_PAD_SINK].flags = MEDIA_PAD_FL_SINK;
    isp->pads[ISP_PAD_SOURCE].flags = MEDIA_PAD_FL_SOURCE;
    isp->subdev.entity.function = MEDIA_ENT_F_PROC_VIDEO_ISP;
    
    ret = media_entity_pads_init(&isp->subdev.entity, ISP_PAD_MAX,
                                isp->pads);
    if (ret)
        goto err_ctrl;
    
    /* Register subdev */
    ret = v4l2_async_register_subdev(&isp->subdev);
    if (ret)
        goto err_entity;
    
    platform_set_drvdata(pdev, isp);
    
    dev_info(&pdev->dev, "ISP registered successfully\n");
    
    return 0;

err_entity:
    media_entity_cleanup(&isp->subdev.entity);
err_ctrl:
    v4l2_ctrl_handler_free(&isp->ctrl_handler);
err_clk:
    clk_disable_unprepare(isp->clk);
    return ret;
}

static const struct of_device_id isp_of_match[] = {
    { .compatible = "vendor,custom-isp" },
    { /* sentinel */ }
};
MODULE_DEVICE_TABLE(of, isp_of_match);

static struct platform_driver isp_driver = {
    .probe = isp_probe,
    .driver = {
        .name = ISP_NAME,
        .of_match_table = isp_of_match,
    },
};

module_platform_driver(isp_driver);

MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Custom ISP Driver");
MODULE_LICENSE("GPL v2");
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: CCM Calibration

**Objective:** Calibrate color correction matrix using color checker chart.

**Procedure:**

1. **Capture color checker:**
   ```bash
   v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=RGGB
   v4l2-ctl --stream-mmap --stream-to=colorchecker.raw --stream-count=1
   ```

2. **Extract patches:**
   ```python
   # extract_patches.py
   import numpy as np
   import cv2
   
   # Load RAW and convert to RGB
   raw = np.fromfile('colorchecker.raw', dtype=np.uint16)
   raw = raw.reshape((1080, 1920))
   rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2RGB)
   
   # Define patch locations (24 patches)
   patches = []
   for i in range(4):
       for j in range(6):
           x = 100 + j * 300
           y = 100 + i * 240
           patch = rgb[y:y+200, x:x+200]
           avg_color = np.mean(patch, axis=(0,1))
           patches.append(avg_color)
   
   np.save('measured_patches.npy', patches)
   ```

3. **Calculate CCM:**
   ```python
   # calculate_ccm.py
   import numpy as np
   
   # Reference colors (from Macbeth chart spec)
   reference = np.load('reference_patches.npy')
   measured = np.load('measured_patches.npy')
   
   # Solve least-squares: measured × CCM = reference
   ccm, residuals, rank, s = np.linalg.lstsq(measured, reference, rcond=None)
   
   print("Color Correction Matrix:")
   print(ccm)
   ```

### Lab 2: Gamma Curve Analysis

**Tool:**

```python
#!/usr/bin/env python3
"""
gamma_analysis.py
Analyze and visualize gamma curves
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_gamma_curves():
    x = np.linspace(0, 1, 256)
    
    plt.figure(figsize=(10, 6))
    
    # Different gamma values
    for gamma in [1.0, 1.8, 2.2, 2.4]:
        y = np.power(x, 1.0/gamma)
        plt.plot(x, y, label=f'γ = {gamma}')
    
    # sRGB
    y_srgb = np.where(x <= 0.0031308,
                     12.92 * x,
                     1.055 * np.power(x, 1/2.4) - 0.055)
    plt.plot(x, y_srgb, label='sRGB', linestyle='--')
    
    plt.xlabel('Linear Input')
    plt.ylabel('Gamma Output')
    plt.title('Gamma Correction Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('gamma_curves.png')
    plt.show()

if __name__ == '__main__':
    plot_gamma_curves()
```

---

## 🐛 Debugging Techniques

### Debug 1: Color Cast Issues

**Symptoms:** Image has unnatural color tint.

**Debug Steps:**

```bash
# Check white balance gains
v4l2-ctl --device=/dev/v4l-subdev1 --get-ctrl=red_balance
v4l2-ctl --device=/dev/v4l-subdev1 --get-ctrl=blue_balance

# Capture gray card image
v4l2-ctl --stream-mmap --stream-to=graycard.raw --stream-count=1

# Analyze with Python
python3 << EOF
import numpy as np
raw = np.fromfile('graycard.raw', dtype=np.uint16).reshape((1080, 1920))

# Calculate channel averages
r_avg = np.mean(raw[0::2, 0::2])
g_avg = np.mean(raw[0::2, 1::2]) + np.mean(raw[1::2, 0::2])
g_avg /= 2
b_avg = np.mean(raw[1::2, 1::2])

print(f"R: {r_avg:.1f}, G: {g_avg:.1f}, B: {b_avg:.1f}")
print(f"Suggested R gain: {g_avg/r_avg:.3f}")
print(f"Suggested B gain: {g_avg/b_avg:.3f}")
EOF
```

### Debug 2: Vignetting/Shading

**Analysis Tool:**

```python
#!/usr/bin/env python3
"""
analyze_shading.py
Visualize lens shading
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_shading(filename, width, height):
    raw = np.fromfile(filename, dtype=np.uint16)
    raw = raw.reshape((height, width))
    
    # Extract green channel
    g = (raw[0::2, 1::2] + raw[1::2, 0::2]) / 2
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(0, g.shape[1])
    y = np.arange(0, g.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, g, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('Lens Shading Profile')
    
    plt.colorbar(surf)
    plt.savefig('shading_profile.png')
    plt.show()

if __name__ == '__main__':
    analyze_shading('uniform_scene.raw', 1920, 1080)
```

---

## ⚡ Performance Optimization

### Optimization 1: SIMD Acceleration

```c
/**
 * @brief SIMD-optimized RGB to YUV conversion (ARM NEON)
 */
#ifdef __ARM_NEON
#include <arm_neon.h>

static void rgb_to_yuv_neon(
    uint8_t *rgb,
    uint8_t *y,
    uint8_t *u,
    uint8_t *v,
    unsigned int pixels)
{
    /* Process 8 pixels at a time */
    unsigned int i;
    for (i = 0; i + 8 <= pixels; i += 8) {
        /* Load RGB */
        uint8x8x3_t rgb_vec = vld3_u8(rgb + i * 3);
        
        /* Convert to 16-bit for arithmetic */
        uint16x8_t r = vmovl_u8(rgb_vec.val[0]);
        uint16x8_t g = vmovl_u8(rgb_vec.val[1]);
        uint16x8_t b = vmovl_u8(rgb_vec.val[2]);
        
        /* Y = 0.299R + 0.587G + 0.114B */
        uint16x8_t y_vec = vmulq_n_u16(r, 77);   /* 0.299 * 256 */
        y_vec = vmlaq_n_u16(y_vec, g, 150);      /* 0.587 * 256 */
        y_vec = vmlaq_n_u16(y_vec, b, 29);       /* 0.114 * 256 */
        y_vec = vshrq_n_u16(y_vec, 8);
        
        /* Store Y */
        vst1_u8(y + i, vmovn_u16(y_vec));
        
        /* U and V calculations similar */
    }
    
    /* Handle remaining pixels */
    for (; i < pixels; i++) {
        rgb_to_yuv_bt601(rgb[i*3], rgb[i*3+1], rgb[i*3+2],
                        &y[i], &u[i], &v[i]);
    }
}
#endif
```

### Optimization 2: Hardware Acceleration

```c
/**
 * @brief Use DMA for ISP buffer transfers
 */
static int isp_setup_dma(struct isp_device *isp)
{
    struct dma_chan *chan;
    struct dma_slave_config config = {0};
    
    chan = dma_request_chan(isp->dev, "isp-dma");
    if (IS_ERR(chan))
        return PTR_ERR(chan);
    
    config.direction = DMA_MEM_TO_DEV;
    config.dst_addr = isp->phys_base + ISP_INPUT_FIFO;
    config.dst_addr_width = DMA_SLAVE_BUSWIDTH_4_BYTES;
    config.dst_maxburst = 16;
    
    dmaengine_slave_config(chan, &config);
    
    isp->dma_chan = chan;
    
    return 0;
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain why lens shading correction is necessary. What causes vignetting?**

2. **What is the purpose of the color correction matrix? Why can't we use sensor RGB directly?**

3. **Describe the difference between gamma correction and tone mapping.**

4. **Why is YUV420 more efficient than RGB24 for video encoding?**

5. **Calculate memory bandwidth for 4K@60fps ISP:**
   - Input: RAW12
   - Output: YUV420

### Practical Challenges

1. **Implement a histogram-based auto white balance algorithm.**

2. **Design an ISP pipeline that minimizes memory bandwidth.**

3. **Debug an image with severe color fringing after demosaicing.**

4. **Optimize ISP processing time from 50ms to < 16ms per frame.**

---

## 📚 Further Reading & Resources

### Standards
- [ITU-R BT.601](https://www.itu.int/rec/R-REC-BT.601/) - Color space conversion
- [ITU-R BT.709](https://www.itu.int/rec/R-REC-BT.709/) - HDTV standard
- [sRGB Specification](https://www.w3.org/Graphics/Color/sRGB)

### Books
- "Digital Image Processing" - Gonzalez & Woods
- "Color Imaging: Fundamentals and Applications" - Reinhard et al.

### Tools
- **ImageMagick:** Image processing and analysis
- **RawTherapee:** RAW image development
- **dcraw:** RAW image decoder

---

## 🎓 Summary

Today we covered:
- ✅ Complete ISP pipeline architecture
- ✅ Black level and lens shading correction
- ✅ White balance and color correction matrix
- ✅ Gamma correction and tone mapping
- ✅ RGB to YUV color space conversion
- ✅ Software and hardware ISP implementations

**Key Takeaways:**
1. ISP pipeline transforms RAW sensor data to display-ready images
2. Each stage addresses specific sensor/optical limitations
3. Color science is critical for accurate reproduction
4. Performance optimization requires hardware acceleration

**Next:** Day 6 - Advanced Image Processing and Noise Reduction

---

**Day 5 Complete** | Phase 3: Camera Systems & ISP | Week 1: Camera Fundamentals
