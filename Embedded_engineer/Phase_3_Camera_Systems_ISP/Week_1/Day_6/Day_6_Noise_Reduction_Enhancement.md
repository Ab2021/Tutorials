# Day 6: Advanced Noise Reduction and Image Enhancement
## Phase 3: Camera Systems & ISP | Week 1: Camera Fundamentals

---

## 🎯 Learning Objectives
1. **Understand** noise sources and characteristics in image sensors
2. **Implement** spatial and temporal noise reduction algorithms
3. **Configure** edge-preserving filters (bilateral, non-local means)
4. **Develop** sharpening and detail enhancement techniques
5. **Debug** noise reduction artifacts and over-processing
6. **Optimize** real-time noise reduction performance

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera system with adjustable gain/exposure
*   **Software:** Image processing libraries, CUDA/OpenCL (optional)
*   **Knowledge:** Signal processing, statistics, convolution
*   **Tools:** Noise measurement tools, test patterns

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Noise Characteristics

#### 1.1 Noise Sources

**Types of Noise:**

1. **Shot Noise (Photon Noise):**
   - Quantum nature of light
   - Poisson distribution
   - σ_shot = √N (N = photon count)
   - Unavoidable, fundamental limit

2. **Read Noise:**
   - Sensor readout circuitry
   - Gaussian distribution
   - Typically 1-5 electrons RMS
   - Dominant in dark scenes

3. **Dark Current Noise:**
   - Thermal generation of electrons
   - Increases with temperature
   - Doubles every ~8°C
   - σ_dark = √(dark_current × exposure_time)

4. **Fixed Pattern Noise (FPN):**
   - Pixel-to-pixel variation
   - Removed by calibration
   - Includes PRNU (Photo Response Non-Uniformity)

**Signal-to-Noise Ratio (SNR):**

```c
/**
 * @brief Calculate SNR for image sensor
 */
struct noise_model {
    float shot_noise;      /* √signal */
    float read_noise;      /* Constant, electrons RMS */
    float dark_current;    /* e-/pixel/second */
    float exposure_time;   /* seconds */
};

static float calculate_snr(
    float signal_electrons,
    struct noise_model *noise)
{
    /* Total noise variance */
    float shot_var = signal_electrons;  /* Poisson */
    float read_var = noise->read_noise * noise->read_noise;
    float dark_var = noise->dark_current * noise->exposure_time;
    
    float total_noise = sqrtf(shot_var + read_var + dark_var);
    
    /* SNR in dB */
    float snr_db = 20.0f * log10f(signal_electrons / total_noise);
    
    return snr_db;
}

/* Example: 10,000 electrons signal */
/* Shot noise: √10000 = 100 e- */
/* Read noise: 2 e- */
/* Dark: 0.1 e-/s × 0.01s = 0.001 e- */
/* Total noise: √(10000 + 4 + 0.001) ≈ 100 e- */
/* SNR = 20×log10(10000/100) = 40 dB */
```

#### 1.2 Noise Measurement

**Temporal Noise (from static scene):**

```c
/**
 * @brief Measure temporal noise from frame sequence
 */
static float measure_temporal_noise(
    uint16_t **frames,
    unsigned int num_frames,
    unsigned int width,
    unsigned int height)
{
    unsigned int total_pixels = width * height;
    float *variance = calloc(total_pixels, sizeof(float));
    
    /* Calculate per-pixel variance across frames */
    for (unsigned int p = 0; p < total_pixels; p++) {
        /* Calculate mean */
        float mean = 0.0f;
        for (unsigned int f = 0; f < num_frames; f++) {
            mean += frames[f][p];
        }
        mean /= num_frames;
        
        /* Calculate variance */
        float var = 0.0f;
        for (unsigned int f = 0; f < num_frames; f++) {
            float diff = frames[f][p] - mean;
            var += diff * diff;
        }
        variance[p] = var / (num_frames - 1);
    }
    
    /* Average variance across all pixels */
    float avg_variance = 0.0f;
    for (unsigned int p = 0; p < total_pixels; p++) {
        avg_variance += variance[p];
    }
    avg_variance /= total_pixels;
    
    float noise_std = sqrtf(avg_variance);
    
    free(variance);
    return noise_std;
}
```

**Spatial Noise (from uniform patch):**

```c
/**
 * @brief Measure spatial noise from uniform region
 */
static float measure_spatial_noise(
    uint16_t *image,
    unsigned int x, unsigned int y,
    unsigned int patch_width, unsigned int patch_height,
    unsigned int image_width)
{
    unsigned int patch_pixels = patch_width * patch_height;
    
    /* Calculate mean */
    float mean = 0.0f;
    for (unsigned int j = 0; j < patch_height; j++) {
        for (unsigned int i = 0; i < patch_width; i++) {
            unsigned int idx = (y + j) * image_width + (x + i);
            mean += image[idx];
        }
    }
    mean /= patch_pixels;
    
    /* Calculate standard deviation */
    float variance = 0.0f;
    for (unsigned int j = 0; j < patch_height; j++) {
        for (unsigned int i = 0; i < patch_width; i++) {
            unsigned int idx = (y + j) * image_width + (x + i);
            float diff = image[idx] - mean;
            variance += diff * diff;
        }
    }
    variance /= (patch_pixels - 1);
    
    return sqrtf(variance);
}
```

### 🔹 Part 2: Spatial Noise Reduction

#### 2.1 Gaussian Blur

**Basic Convolution:**

```c
/**
 * @brief 2D Gaussian kernel generation
 */
static void generate_gaussian_kernel(
    float *kernel,
    unsigned int size,
    float sigma)
{
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;
            float value = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    /* Normalize */
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

/**
 * @brief Apply Gaussian blur
 */
static void gaussian_blur(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float sigma)
{
    const unsigned int kernel_size = 5;
    float *kernel = malloc(kernel_size * kernel_size * sizeof(float));
    generate_gaussian_kernel(kernel, kernel_size, sigma);
    
    int radius = kernel_size / 2;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    
                    /* Boundary handling */
                    px = (px < 0) ? 0 : ((px >= width) ? width-1 : px);
                    py = (py < 0) ? 0 : ((py >= height) ? height-1 : py);
                    
                    unsigned int k_idx = (ky + radius) * kernel_size + (kx + radius);
                    unsigned int p_idx = py * width + px;
                    
                    sum += input[p_idx] * kernel[k_idx];
                }
            }
            
            output[y * width + x] = (uint16_t)sum;
        }
    }
    
    free(kernel);
}
```

#### 2.2 Bilateral Filter

**Edge-Preserving Denoising:**

```c
/**
 * @brief Bilateral filter - preserves edges while reducing noise
 */
static void bilateral_filter(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float sigma_spatial,
    float sigma_range)
{
    int radius = (int)(3.0f * sigma_spatial);
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int center_idx = y * width + x;
            uint16_t center_val = input[center_idx];
            
            float sum_weights = 0.0f;
            float sum_values = 0.0f;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int px = x + dx;
                    int py = y + dy;
                    
                    /* Boundary check */
                    if (px < 0 || px >= width || py < 0 || py >= height)
                        continue;
                    
                    unsigned int p_idx = py * width + px;
                    uint16_t p_val = input[p_idx];
                    
                    /* Spatial weight (Gaussian based on distance) */
                    float spatial_dist = sqrtf(dx*dx + dy*dy);
                    float spatial_weight = expf(-(spatial_dist * spatial_dist) /
                                               (2.0f * sigma_spatial * sigma_spatial));
                    
                    /* Range weight (Gaussian based on intensity difference) */
                    float range_dist = abs(p_val - center_val);
                    float range_weight = expf(-(range_dist * range_dist) /
                                             (2.0f * sigma_range * sigma_range));
                    
                    /* Combined weight */
                    float weight = spatial_weight * range_weight;
                    
                    sum_weights += weight;
                    sum_values += weight * p_val;
                }
            }
            
            output[center_idx] = (uint16_t)(sum_values / sum_weights);
        }
    }
}
```

#### 2.3 Non-Local Means (NLM)

**Patch-Based Denoising:**

```c
/**
 * @brief Non-local means denoising
 */
#define NLM_PATCH_SIZE 7
#define NLM_SEARCH_WINDOW 21

static float compute_patch_distance(
    uint16_t *img,
    unsigned int width,
    unsigned int x1, unsigned int y1,
    unsigned int x2, unsigned int y2,
    unsigned int patch_size)
{
    int radius = patch_size / 2;
    float dist = 0.0f;
    unsigned int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px1 = x1 + dx;
            int py1 = y1 + dy;
            int px2 = x2 + dx;
            int py2 = y2 + dy;
            
            if (px1 < 0 || px1 >= width || py1 < 0 ||
                px2 < 0 || px2 >= width || py2 < 0)
                continue;
            
            unsigned int idx1 = py1 * width + px1;
            unsigned int idx2 = py2 * width + px2;
            
            float diff = img[idx1] - img[idx2];
            dist += diff * diff;
            count++;
        }
    }
    
    return dist / count;
}

static void non_local_means(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float h)  /* Filtering parameter */
{
    int search_radius = NLM_SEARCH_WINDOW / 2;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            float sum_weights = 0.0f;
            float sum_values = 0.0f;
            
            /* Search window */
            for (int dy = -search_radius; dy <= search_radius; dy++) {
                for (int dx = -search_radius; dx <= search_radius; dx++) {
                    int px = x + dx;
                    int py = y + dy;
                    
                    if (px < 0 || px >= width || py < 0 || py >= height)
                        continue;
                    
                    /* Compute patch distance */
                    float dist = compute_patch_distance(input, width,
                                                        x, y, px, py,
                                                        NLM_PATCH_SIZE);
                    
                    /* Weight based on patch similarity */
                    float weight = expf(-dist / (h * h));
                    
                    sum_weights += weight;
                    sum_values += weight * input[py * width + px];
                }
            }
            
            output[y * width + x] = (uint16_t)(sum_values / sum_weights);
        }
    }
}
```

### 🔹 Part 3: Temporal Noise Reduction

#### 3.1 Frame Averaging

**Simple Temporal Filter:**

```c
/**
 * @brief Temporal noise reduction using exponential moving average
 */
struct temporal_nr_context {
    uint16_t *history;  /* Previous frame */
    unsigned int width;
    unsigned int height;
    float alpha;        /* Blending factor (0-1) */
};

static void temporal_nr_init(
    struct temporal_nr_context *ctx,
    unsigned int width,
    unsigned int height,
    float alpha)
{
    ctx->width = width;
    ctx->height = height;
    ctx->alpha = alpha;
    ctx->history = calloc(width * height, sizeof(uint16_t));
}

static void temporal_nr_process(
    struct temporal_nr_context *ctx,
    uint16_t *current,
    uint16_t *output)
{
    unsigned int total_pixels = ctx->width * ctx->height;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        /* Exponential moving average */
        float filtered = ctx->alpha * current[i] + 
                        (1.0f - ctx->alpha) * ctx->history[i];
        
        output[i] = (uint16_t)filtered;
        ctx->history[i] = output[i];
    }
}
```

#### 3.2 Motion-Adaptive Temporal NR

**Detect Motion and Adjust Filtering:**

```c
/**
 * @brief Motion-adaptive temporal noise reduction
 */
struct motion_adaptive_tnr {
    uint16_t *prev_frame;
    uint16_t *motion_map;
    unsigned int width;
    unsigned int height;
    uint16_t motion_threshold;
};

static void compute_motion_map(
    struct motion_adaptive_tnr *tnr,
    uint16_t *current)
{
    unsigned int total_pixels = tnr->width * tnr->height;
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        uint16_t diff = abs(current[i] - tnr->prev_frame[i]);
        tnr->motion_map[i] = diff;
    }
}

static void motion_adaptive_tnr_process(
    struct motion_adaptive_tnr *tnr,
    uint16_t *current,
    uint16_t *output)
{
    unsigned int total_pixels = tnr->width * tnr->height;
    
    /* Compute motion */
    compute_motion_map(tnr, current);
    
    for (unsigned int i = 0; i < total_pixels; i++) {
        float alpha;
        
        if (tnr->motion_map[i] < tnr->motion_threshold) {
            /* Low motion: strong temporal filtering */
            alpha = 0.2f;
        } else {
            /* High motion: weak temporal filtering (avoid ghosting) */
            alpha = 0.8f;
        }
        
        output[i] = (uint16_t)(alpha * current[i] + 
                              (1.0f - alpha) * tnr->prev_frame[i]);
    }
    
    /* Update history */
    memcpy(tnr->prev_frame, output, total_pixels * sizeof(uint16_t));
}
```

### 🔹 Part 4: Sharpening and Detail Enhancement

#### 4.1 Unsharp Masking

**Classic Sharpening Technique:**

```c
/**
 * @brief Unsharp masking for sharpening
 */
static void unsharp_mask(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float amount,
    float sigma)
{
    /* 1. Create blurred version */
    uint16_t *blurred = malloc(width * height * sizeof(uint16_t));
    gaussian_blur(input, blurred, width, height, sigma);
    
    /* 2. Subtract blurred from original to get high-frequency details */
    /* 3. Add scaled details back to original */
    for (unsigned int i = 0; i < width * height; i++) {
        int detail = input[i] - blurred[i];
        int sharpened = input[i] + (int)(amount * detail);
        
        /* Clamp */
        if (sharpened < 0)
            sharpened = 0;
        else if (sharpened > 1023)
            sharpened = 1023;
        
        output[i] = (uint16_t)sharpened;
    }
    
    free(blurred);
}
```

#### 4.2 Laplacian Sharpening

**Edge Detection Based:**

```c
/**
 * @brief Laplacian sharpening
 */
static void laplacian_sharpen(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float strength)
{
    /* Laplacian kernel */
    const int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  4, -1},
        { 0, -1,  0}
    };
    
    for (unsigned int y = 1; y < height - 1; y++) {
        for (unsigned int x = 1; x < width - 1; x++) {
            int laplacian = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    unsigned int idx = (y + ky) * width + (x + kx);
                    laplacian += input[idx] * kernel[ky + 1][kx + 1];
                }
            }
            
            unsigned int center_idx = y * width + x;
            int sharpened = input[center_idx] + (int)(strength * laplacian);
            
            /* Clamp */
            if (sharpened < 0)
                sharpened = 0;
            else if (sharpened > 1023)
                sharpened = 1023;
            
            output[center_idx] = (uint16_t)sharpened;
        }
    }
}
```

#### 4.3 Adaptive Sharpening

**Sharpen Based on Local Contrast:**

```c
/**
 * @brief Adaptive sharpening based on local variance
 */
static void adaptive_sharpen(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float max_strength)
{
    const int window_size = 5;
    int radius = window_size / 2;
    
    for (unsigned int y = radius; y < height - radius; y++) {
        for (unsigned int x = radius; x < width - radius; x++) {
            /* Calculate local variance */
            float mean = 0.0f;
            unsigned int count = 0;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    unsigned int idx = (y + dy) * width + (x + dx);
                    mean += input[idx];
                    count++;
                }
            }
            mean /= count;
            
            float variance = 0.0f;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    unsigned int idx = (y + dy) * width + (x + dx);
                    float diff = input[idx] - mean;
                    variance += diff * diff;
                }
            }
            variance /= count;
            
            /* Adaptive strength based on variance */
            /* High variance (edges) -> more sharpening */
            /* Low variance (flat) -> less sharpening */
            float strength = max_strength * (variance / (variance + 100.0f));
            
            /* Apply unsharp mask with adaptive strength */
            unsigned int center_idx = y * width + x;
            
            /* Simple 3x3 blur for high-freq extraction */
            float blurred = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    unsigned int idx = (y + dy) * width + (x + dx);
                    blurred += input[idx];
                }
            }
            blurred /= 9.0f;
            
            float detail = input[center_idx] - blurred;
            int sharpened = input[center_idx] + (int)(strength * detail);
            
            /* Clamp */
            output[center_idx] = (sharpened < 0) ? 0 : 
                                ((sharpened > 1023) ? 1023 : sharpened);
        }
    }
}
```

---

## 💻 Implementation Examples

### Example 1: Complete Noise Reduction Pipeline

```c
/**
 * @file nr_pipeline.c
 * @brief Complete noise reduction and enhancement pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

struct nr_pipeline_config {
    /* Spatial NR */
    bool enable_bilateral;
    float bilateral_sigma_spatial;
    float bilateral_sigma_range;
    
    /* Temporal NR */
    bool enable_temporal;
    float temporal_alpha;
    uint16_t motion_threshold;
    
    /* Sharpening */
    bool enable_sharpen;
    float sharpen_amount;
    float sharpen_sigma;
};

struct nr_pipeline {
    struct nr_pipeline_config config;
    
    /* Temporal NR state */
    struct motion_adaptive_tnr tnr;
    
    /* Buffers */
    uint16_t *temp_buffer;
    
    unsigned int width;
    unsigned int height;
};

/**
 * @brief Initialize NR pipeline
 */
static struct nr_pipeline *nr_pipeline_init(
    unsigned int width,
    unsigned int height,
    struct nr_pipeline_config *config)
{
    struct nr_pipeline *nr = calloc(1, sizeof(*nr));
    if (!nr)
        return NULL;
    
    nr->width = width;
    nr->height = height;
    nr->config = *config;
    
    /* Allocate buffers */
    nr->temp_buffer = malloc(width * height * sizeof(uint16_t));
    
    /* Initialize temporal NR */
    if (config->enable_temporal) {
        nr->tnr.width = width;
        nr->tnr.height = height;
        nr->tnr.motion_threshold = config->motion_threshold;
        nr->tnr.prev_frame = calloc(width * height, sizeof(uint16_t));
        nr->tnr.motion_map = malloc(width * height * sizeof(uint16_t));
    }
    
    return nr;
}

/**
 * @brief Process frame through NR pipeline
 */
static void nr_pipeline_process(
    struct nr_pipeline *nr,
    uint16_t *input,
    uint16_t *output)
{
    uint16_t *current = input;
    
    /* 1. Spatial noise reduction */
    if (nr->config.enable_bilateral) {
        printf("Applying bilateral filter...\n");
        bilateral_filter(current, nr->temp_buffer,
                        nr->width, nr->height,
                        nr->config.bilateral_sigma_spatial,
                        nr->config.bilateral_sigma_range);
        current = nr->temp_buffer;
    }
    
    /* 2. Temporal noise reduction */
    if (nr->config.enable_temporal) {
        printf("Applying temporal NR...\n");
        motion_adaptive_tnr_process(&nr->tnr, current, output);
        current = output;
    } else {
        memcpy(output, current, nr->width * nr->height * sizeof(uint16_t));
    }
    
    /* 3. Sharpening */
    if (nr->config.enable_sharpen) {
        printf("Applying sharpening...\n");
        unsharp_mask(current, nr->temp_buffer,
                    nr->width, nr->height,
                    nr->config.sharpen_amount,
                    nr->config.sharpen_sigma);
        memcpy(output, nr->temp_buffer, nr->width * nr->height * sizeof(uint16_t));
    }
}

/**
 * @brief Example usage
 */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.raw> <output.raw>\n", argv[0]);
        return 1;
    }
    
    unsigned int width = 1920;
    unsigned int height = 1080;
    
    /* Configure pipeline */
    struct nr_pipeline_config config = {
        .enable_bilateral = true,
        .bilateral_sigma_spatial = 3.0f,
        .bilateral_sigma_range = 50.0f,
        
        .enable_temporal = true,
        .temporal_alpha = 0.3f,
        .motion_threshold = 20,
        
        .enable_sharpen = true,
        .sharpen_amount = 1.5f,
        .sharpen_sigma = 1.0f,
    };
    
    struct nr_pipeline *nr = nr_pipeline_init(width, height, &config);
    if (!nr) {
        fprintf(stderr, "Failed to initialize NR pipeline\n");
        return 1;
    }
    
    /* Allocate I/O buffers */
    uint16_t *input = malloc(width * height * sizeof(uint16_t));
    uint16_t *output = malloc(width * height * sizeof(uint16_t));
    
    /* Load input */
    FILE *f_in = fopen(argv[1], "rb");
    fread(input, sizeof(uint16_t), width * height, f_in);
    fclose(f_in);
    
    /* Process */
    nr_pipeline_process(nr, input, output);
    
    /* Save output */
    FILE *f_out = fopen(argv[2], "wb");
    fwrite(output, sizeof(uint16_t), width * height, f_out);
    fclose(f_out);
    
    printf("Processing complete\n");
    
    /* Cleanup */
    free(input);
    free(output);
    free(nr->temp_buffer);
    if (nr->tnr.prev_frame) free(nr->tnr.prev_frame);
    if (nr->tnr.motion_map) free(nr->tnr.motion_map);
    free(nr);
    
    return 0;
}
```

### Example 2: GPU-Accelerated Bilateral Filter (CUDA)

```cuda
/**
 * @file bilateral_cuda.cu
 * @brief CUDA-accelerated bilateral filter
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void bilateral_filter_kernel(
    const uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float sigma_spatial,
    float sigma_range,
    int radius)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    unsigned int center_idx = y * width + x;
    uint16_t center_val = input[center_idx];
    
    float sum_weights = 0.0f;
    float sum_values = 0.0f;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int px = x + dx;
            int py = y + dy;
            
            /* Boundary check */
            if (px < 0 || px >= width || py < 0 || py >= height)
                continue;
            
            unsigned int p_idx = py * width + px;
            uint16_t p_val = input[p_idx];
            
            /* Spatial weight */
            float spatial_dist = sqrtf(dx*dx + dy*dy);
            float spatial_weight = expf(-(spatial_dist * spatial_dist) /
                                       (2.0f * sigma_spatial * sigma_spatial));
            
            /* Range weight */
            float range_dist = abs((int)p_val - (int)center_val);
            float range_weight = expf(-(range_dist * range_dist) /
                                     (2.0f * sigma_range * sigma_range));
            
            float weight = spatial_weight * range_weight;
            
            sum_weights += weight;
            sum_values += weight * p_val;
        }
    }
    
    output[center_idx] = (uint16_t)(sum_values / sum_weights);
}

/**
 * @brief Host function to launch bilateral filter
 */
extern "C" void bilateral_filter_cuda(
    uint16_t *h_input,
    uint16_t *h_output,
    unsigned int width,
    unsigned int height,
    float sigma_spatial,
    float sigma_range)
{
    size_t image_size = width * height * sizeof(uint16_t);
    
    /* Allocate device memory */
    uint16_t *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    
    /* Copy input to device */
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    
    /* Launch kernel */
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    int radius = (int)(3.0f * sigma_spatial);
    
    bilateral_filter_kernel<<<grid_size, block_size>>>(
        d_input, d_output, width, height,
        sigma_spatial, sigma_range, radius
    );
    
    /* Copy result back to host */
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
    
    /* Cleanup */
    cudaFree(d_input);
    cudaFree(d_output);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Noise Characterization

**Objective:** Measure and characterize sensor noise.

**Procedure:**

```bash
#!/bin/bash
# noise_characterization.sh

echo "Sensor Noise Characterization"
echo "=============================="

# Capture dark frames (lens cap on)
echo "Capturing dark frames..."
for gain in 1 2 4 8 16; do
    v4l2-ctl --set-ctrl=gain=$gain
    v4l2-ctl --stream-mmap --stream-to=dark_gain${gain}.raw --stream-count=10
done

# Capture uniform illumination frames
echo "Capturing uniform frames (use gray card)..."
for gain in 1 2 4 8 16; do
    v4l2-ctl --set-ctrl=gain=$gain
    v4l2-ctl --stream-mmap --stream-to=uniform_gain${gain}.raw --stream-count=10
done

# Analyze with Python
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

gains = [1, 2, 4, 8, 16]
read_noise = []
shot_noise = []

for gain in gains:
    # Load dark frames
    dark_frames = []
    for i in range(10):
        frame = np.fromfile(f'dark_gain{gain}.raw', dtype=np.uint16)
        dark_frames.append(frame)
    
    # Calculate temporal std (read noise)
    dark_std = np.std(dark_frames, axis=0)
    read_noise.append(np.mean(dark_std))
    
    # Load uniform frames
    uniform_frames = []
    for i in range(10):
        frame = np.fromfile(f'uniform_gain{gain}.raw', dtype=np.uint16)
        uniform_frames.append(frame)
    
    # Calculate temporal std (read + shot noise)
    uniform_std = np.std(uniform_frames, axis=0)
    total_noise = np.mean(uniform_std)
    
    # Shot noise = sqrt(total^2 - read^2)
    shot = np.sqrt(total_noise**2 - read_noise[-1]**2)
    shot_noise.append(shot)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(gains, read_noise, 'o-', label='Read Noise')
plt.plot(gains, shot_noise, 's-', label='Shot Noise')
plt.xlabel('Gain')
plt.ylabel('Noise (DN)')
plt.title('Noise vs Gain')
plt.legend()
plt.grid(True)
plt.savefig('noise_characterization.png')
plt.show()
EOF
```

### Lab 2: NR Algorithm Comparison

**Test Script:**

```python
#!/usr/bin/env python3
"""
compare_nr_algorithms.py
Compare different noise reduction algorithms
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time

def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def test_algorithm(name, func, clean, noisy):
    """Test NR algorithm and measure performance"""
    print(f"\nTesting {name}...")
    
    start_time = time.time()
    denoised = func(noisy)
    elapsed = time.time() - start_time
    
    psnr_val = psnr(clean, denoised)
    ssim_val = ssim(clean, denoised)
    
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  Time: {elapsed*1000:.1f} ms")
    
    return denoised, psnr_val, ssim_val, elapsed

# Load test image
clean = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)

# Add noise
noise_sigma = 25
noisy = add_gaussian_noise(clean, noise_sigma)

print(f"Original PSNR: {psnr(clean, noisy):.2f} dB")

# Test algorithms
results = {}

# Gaussian blur
results['Gaussian'] = test_algorithm(
    'Gaussian Blur',
    lambda img: cv2.GaussianBlur(img, (5, 5), 1.5),
    clean, noisy
)

# Bilateral filter
results['Bilateral'] = test_algorithm(
    'Bilateral Filter',
    lambda img: cv2.bilateralFilter(img, 9, 75, 75),
    clean, noisy
)

# Non-local means
results['NLM'] = test_algorithm(
    'Non-Local Means',
    lambda img: cv2.fastNlMeansDenoising(img, None, 10, 7, 21),
    clean, noisy
)

# Save comparison
comparison = np.hstack([clean, noisy, 
                       results['Gaussian'][0],
                       results['Bilateral'][0],
                       results['NLM'][0]])
cv2.imwrite('nr_comparison.png', comparison)
```

---

## 🐛 Debugging Techniques

### Debug 1: Over-Smoothing

**Symptoms:** Loss of detail, blurry images.

**Analysis:**

```python
#!/usr/bin/env python3
"""
detect_oversmoothing.py
Detect loss of detail from over-smoothing
"""

import numpy as np
import cv2

def measure_sharpness(image):
    """Measure image sharpness using Laplacian variance"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def analyze_nr_strength(image, nr_func, strengths):
    """Analyze NR at different strengths"""
    results = []
    
    for strength in strengths:
        denoised = nr_func(image, strength)
        sharpness = measure_sharpness(denoised)
        results.append((strength, sharpness))
    
    return results

# Example
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# Test bilateral filter at different sigmas
sigmas = [1, 3, 5, 7, 9, 11]
results = analyze_nr_strength(
    image,
    lambda img, s: cv2.bilateralFilter(img, 9, s*10, s*10),
    sigmas
)

print("Sigma | Sharpness")
print("------|----------")
for sigma, sharpness in results:
    print(f"{sigma:5.1f} | {sharpness:8.2f}")
```

### Debug 2: Temporal Ghosting

**Detection:**

```c
/**
 * @brief Detect temporal ghosting artifacts
 */
static float detect_ghosting(
    uint16_t *frame1,
    uint16_t *frame2,
    uint16_t *frame3,
    unsigned int width,
    unsigned int height)
{
    float ghost_score = 0.0f;
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < width * height; i++) {
        /* Check for temporal inconsistency */
        int diff12 = abs(frame2[i] - frame1[i]);
        int diff23 = abs(frame3[i] - frame2[i]);
        
        /* Ghosting: large change followed by slow decay */
        if (diff12 > 50 && diff23 > 0 && diff23 < diff12 / 2) {
            ghost_score += 1.0f;
            count++;
        }
    }
    
    return count > 0 ? ghost_score / count : 0.0f;
}
```

---

## ⚡ Performance Optimization

### Optimization 1: Separable Filters

```c
/**
 * @brief Separable Gaussian blur (faster than 2D)
 */
static void gaussian_blur_separable(
    uint16_t *input,
    uint16_t *output,
    unsigned int width,
    unsigned int height,
    float sigma)
{
    /* 1D Gaussian kernel */
    const int kernel_size = 5;
    float kernel[kernel_size];
    generate_gaussian_kernel_1d(kernel, kernel_size, sigma);
    
    uint16_t *temp = malloc(width * height * sizeof(uint16_t));
    
    /* Horizontal pass */
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int px = x + k - kernel_size/2;
                px = (px < 0) ? 0 : ((px >= width) ? width-1 : px);
                sum += input[y * width + px] * kernel[k];
            }
            
            temp[y * width + x] = (uint16_t)sum;
        }
    }
    
    /* Vertical pass */
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int py = y + k - kernel_size/2;
                py = (py < 0) ? 0 : ((py >= height) ? height-1 : py);
                sum += temp[py * width + x] * kernel[k];
            }
            
            output[y * width + x] = (uint16_t)sum;
        }
    }
    
    free(temp);
}
/* Complexity: O(width × height × kernel_size) instead of O(width × height × kernel_size^2) */
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the difference between shot noise and read noise. Which dominates in low-light conditions?**

2. **Why does bilateral filter preserve edges better than Gaussian blur?**

3. **What causes temporal ghosting in temporal NR? How can it be prevented?**

4. **Compare spatial vs temporal noise reduction: advantages and disadvantages of each.**

5. **Calculate SNR for:**
   - Signal: 5000 electrons
   - Read noise: 3 e- RMS
   - Dark current: 0.1 e-/s, exposure 10ms

### Practical Challenges

1. **Implement a real-time bilateral filter that runs at 30fps for 1080p.**

2. **Design a motion-adaptive temporal NR that handles fast-moving objects without ghosting.**

3. **Debug an image that shows ringing artifacts after sharpening.**

4. **Optimize NLM denoising to reduce processing time by 50%.**

---

## 📚 Further Reading & Resources

### Papers
- "A Non-Local Algorithm for Image Denoising" - Buades et al.
- "Bilateral Filtering for Gray and Color Images" - Tomasi & Manduchi
- "Noise Reduction in Digital Cameras" - Foi et al.

### Books
- "Image Processing: The Fundamentals" - Petrou & Petrou
- "Handbook of Image and Video Processing" - Bovik

### Tools
- **ImageJ/Fiji:** Noise analysis plugins
- **MATLAB Image Processing Toolbox**
- **OpenCV:** cv::fastNlMeansDenoising

---

## 🎓 Summary

Today we covered:
- ✅ Noise sources and characterization in image sensors
- ✅ Spatial NR: Gaussian, bilateral, non-local means
- ✅ Temporal NR: motion-adaptive filtering
- ✅ Sharpening: unsharp mask, Laplacian, adaptive
- ✅ Complete NR pipeline implementation
- ✅ GPU acceleration with CUDA

**Key Takeaways:**
1. Different noise types require different reduction strategies
2. Edge-preserving filters are critical for maintaining image quality
3. Temporal NR is powerful but must handle motion carefully
4. Sharpening should be adaptive to avoid artifacts

**Next:** Day 7 - Week 1 Review and Camera System Integration Project

---

**Day 6 Complete** | Phase 3: Camera Systems & ISP | Week 1: Camera Fundamentals
