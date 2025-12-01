# Day 12: Advanced 3A Algorithms - Auto-Focus
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** auto-focus principles and contrast/phase detection methods
2. **Implement** focus search algorithms (hill climbing, binary search)
3. **Configure** focus metrics and sharpness measurement
4. **Develop** continuous auto-focus (CAF) for video
5. **Debug** focus hunting and convergence issues
6. **Optimize** focus speed and accuracy

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with motorized lens or VCM (Voice Coil Motor)
*   **Software:** Focus control drivers, image processing libraries
*   **Knowledge:** Image sharpness metrics, control algorithms
*   **Tools:** Focus test charts, measurement equipment

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Auto-Focus Fundamentals

#### 1.1 Focus Metrics

**Contrast-Based Metrics:**

```c
/**
 * @brief Focus sharpness metrics
 */

/* 1. Variance-based (simple, fast) */
static float focus_metric_variance(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int roi_x,
    unsigned int roi_y,
    unsigned int roi_width,
    unsigned int roi_height)
{
    uint64_t sum = 0;
    uint64_t sum_sq = 0;
    unsigned int count = 0;
    
    for (unsigned int y = roi_y; y < roi_y + roi_height; y++) {
        for (unsigned int x = roi_x; x < roi_x + roi_width; x++) {
            uint8_t val = image[y * width + x];
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    
    float mean = (float)sum / count;
    float variance = ((float)sum_sq / count) - (mean * mean);
    
    return variance;
}

/* 2. Laplacian-based (edge detection) */
static float focus_metric_laplacian(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int roi_x,
    unsigned int roi_y,
    unsigned int roi_width,
    unsigned int roi_height)
{
    /* Laplacian kernel */
    const int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  4, -1},
        { 0, -1,  0}
    };
    
    float sum = 0.0f;
    
    for (unsigned int y = roi_y + 1; y < roi_y + roi_height - 1; y++) {
        for (unsigned int x = roi_x + 1; x < roi_x + roi_width - 1; x++) {
            int laplacian = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    unsigned int idx = (y + ky) * width + (x + kx);
                    laplacian += image[idx] * kernel[ky + 1][kx + 1];
                }
            }
            
            sum += abs(laplacian);
        }
    }
    
    return sum / ((roi_width - 2) * (roi_height - 2));
}

/* 3. Tenengrad (Sobel-based) */
static float focus_metric_tenengrad(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int roi_x,
    unsigned int roi_y,
    unsigned int roi_width,
    unsigned int roi_height)
{
    float sum = 0.0f;
    
    for (unsigned int y = roi_y + 1; y < roi_y + roi_height - 1; y++) {
        for (unsigned int x = roi_x + 1; x < roi_x + roi_width - 1; x++) {
            /* Sobel X */
            int gx = -image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)]
                    -2*image[y*width + (x-1)] + 2*image[y*width + (x+1)]
                    -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)];
            
            /* Sobel Y */
            int gy = -image[(y-1)*width + (x-1)] - 2*image[(y-1)*width + x] - image[(y-1)*width + (x+1)]
                    +image[(y+1)*width + (x-1)] + 2*image[(y+1)*width + x] + image[(y+1)*width + (x+1)];
            
            float gradient = sqrtf(gx*gx + gy*gy);
            
            /* Threshold to reduce noise sensitivity */
            if (gradient > 10.0f) {
                sum += gradient * gradient;
            }
        }
    }
    
    return sum;
}

/* 4. Frequency-based (FFT magnitude) */
static float focus_metric_frequency(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int roi_x,
    unsigned int roi_y,
    unsigned int roi_width,
    unsigned int roi_height)
{
    /* Extract ROI */
    float *roi = malloc(roi_width * roi_height * sizeof(float));
    
    for (unsigned int y = 0; y < roi_height; y++) {
        for (unsigned int x = 0; x < roi_width; x++) {
            roi[y * roi_width + x] = image[(roi_y + y) * width + (roi_x + x)];
        }
    }
    
    /* Compute 2D FFT */
    /* This would use FFTW or similar library */
    float *fft_magnitude = compute_fft_2d(roi, roi_width, roi_height);
    
    /* Sum high-frequency components */
    float high_freq_sum = 0.0f;
    unsigned int center_x = roi_width / 2;
    unsigned int center_y = roi_height / 2;
    
    for (unsigned int y = 0; y < roi_height; y++) {
        for (unsigned int x = 0; x < roi_width; x++) {
            float dx = x - center_x;
            float dy = y - center_y;
            float freq = sqrtf(dx*dx + dy*dy);
            
            /* High frequency threshold */
            if (freq > roi_width / 4) {
                high_freq_sum += fft_magnitude[y * roi_width + x];
            }
        }
    }
    
    free(roi);
    free(fft_magnitude);
    
    return high_freq_sum;
}
```

#### 1.2 Focus Search Algorithms

**Hill Climbing:**

```c
/**
 * @brief Hill climbing focus search
 */
struct focus_search_state {
    int current_position;
    int min_position;
    int max_position;
    int step_size;
    
    float current_metric;
    float prev_metric;
    
    enum {
        SEARCH_COARSE,
        SEARCH_FINE,
        SEARCH_DONE
    } state;
};

static int focus_search_hill_climbing(
    struct focus_search_state *search,
    float (*metric_func)(void*),
    void *metric_data,
    int (*move_lens)(int position))
{
    switch (search->state) {
    case SEARCH_COARSE:
        /* Coarse search with large steps */
        search->current_position += search->step_size;
        
        if (search->current_position > search->max_position) {
            /* Reached end, reverse direction */
            search->step_size = -search->step_size;
            search->current_position += search->step_size;
        }
        
        move_lens(search->current_position);
        usleep(50000);  /* Wait for lens to settle */
        
        search->current_metric = metric_func(metric_data);
        
        /* Check if we passed the peak */
        if (search->current_metric < search->prev_metric) {
            /* Peak was at previous position, switch to fine search */
            search->current_position -= search->step_size;
            search->step_size = search->step_size / 4;  /* Reduce step */
            search->state = SEARCH_FINE;
        }
        
        search->prev_metric = search->current_metric;
        break;
    
    case SEARCH_FINE:
        /* Fine search around peak */
        search->current_position += search->step_size;
        move_lens(search->current_position);
        usleep(30000);
        
        search->current_metric = metric_func(metric_data);
        
        if (search->current_metric < search->prev_metric) {
            /* Found peak */
            search->current_position -= search->step_size;
            move_lens(search->current_position);
            search->state = SEARCH_DONE;
        }
        
        search->prev_metric = search->current_metric;
        break;
    
    case SEARCH_DONE:
        return 1;  /* Focus complete */
    }
    
    return 0;  /* Continue searching */
}
```

**Binary Search:**

```c
/**
 * @brief Binary search for focus peak
 */
struct binary_focus_search {
    int positions[16];  /* Sample positions */
    float metrics[16];  /* Corresponding metrics */
    unsigned int num_samples;
    
    int left;
    int right;
    int peak_position;
};

static int focus_search_binary(
    struct binary_focus_search *search,
    float (*metric_func)(void*),
    void *metric_data,
    int (*move_lens)(int position))
{
    if (search->num_samples == 0) {
        /* Initialize: sample entire range */
        int range = search->right - search->left;
        int step = range / 15;
        
        for (unsigned int i = 0; i < 16; i++) {
            search->positions[i] = search->left + i * step;
            move_lens(search->positions[i]);
            usleep(50000);
            search->metrics[i] = metric_func(metric_data);
        }
        
        search->num_samples = 16;
        
        /* Find initial peak */
        float max_metric = 0.0f;
        for (unsigned int i = 0; i < 16; i++) {
            if (search->metrics[i] > max_metric) {
                max_metric = search->metrics[i];
                search->peak_position = search->positions[i];
            }
        }
    }
    
    /* Refine around peak */
    int range = (search->right - search->left) / 4;
    
    if (range < 10) {
        /* Converged */
        move_lens(search->peak_position);
        return 1;
    }
    
    search->left = search->peak_position - range;
    search->right = search->peak_position + range;
    search->num_samples = 0;  /* Trigger resampling */
    
    return 0;
}
```

**Fibonacci Search:**

```c
/**
 * @brief Fibonacci search for optimal focus
 */
static int focus_search_fibonacci(
    int min_pos,
    int max_pos,
    float (*metric_func)(void*),
    void *metric_data,
    int (*move_lens)(int position))
{
    /* Generate Fibonacci numbers */
    int fib[20];
    fib[0] = 1;
    fib[1] = 1;
    
    int n = 2;
    while (fib[n-1] < (max_pos - min_pos)) {
        fib[n] = fib[n-1] + fib[n-2];
        n++;
    }
    
    int k = n - 1;
    int x1 = min_pos + fib[k-2];
    int x2 = min_pos + fib[k-1];
    
    move_lens(x1);
    usleep(50000);
    float f1 = metric_func(metric_data);
    
    move_lens(x2);
    usleep(50000);
    float f2 = metric_func(metric_data);
    
    for (int i = k; i > 1; i--) {
        if (f1 > f2) {
            max_pos = x2;
            x2 = x1;
            f2 = f1;
            x1 = min_pos + fib[i-3];
            
            move_lens(x1);
            usleep(50000);
            f1 = metric_func(metric_data);
        } else {
            min_pos = x1;
            x1 = x2;
            f1 = f2;
            x2 = min_pos + fib[i-2];
            
            move_lens(x2);
            usleep(50000);
            f2 = metric_func(metric_data);
        }
    }
    
    return (f1 > f2) ? x1 : x2;
}
```

### 🔹 Part 2: Phase Detection Auto-Focus (PDAF)

#### 2.1 PDAF Principles

**Phase Difference Calculation:**

```c
/**
 * @brief Phase detection auto-focus
 */

/* PDAF pixel pairs */
struct pdaf_pixel_pair {
    uint16_t left;   /* Left-masked pixel */
    uint16_t right;  /* Right-masked pixel */
    unsigned int x, y;
};

/**
 * @brief Calculate phase difference from PDAF pixels
 */
static float calculate_phase_difference(
    struct pdaf_pixel_pair *pairs,
    unsigned int num_pairs)
{
    /* Cross-correlation to find shift */
    float best_correlation = -1.0f;
    int best_shift = 0;
    
    for (int shift = -10; shift <= 10; shift++) {
        float correlation = 0.0f;
        unsigned int count = 0;
        
        for (unsigned int i = 0; i < num_pairs; i++) {
            /* Find corresponding pair with shift */
            int j = i + shift;
            if (j < 0 || j >= num_pairs)
                continue;
            
            correlation += pairs[i].left * pairs[j].right;
            count++;
        }
        
        correlation /= count;
        
        if (correlation > best_correlation) {
            best_correlation = correlation;
            best_shift = shift;
        }
    }
    
    return best_shift;
}

/**
 * @brief Convert phase difference to lens position
 */
static int pdaf_to_lens_position(
    float phase_diff,
    int current_position,
    float calibration_factor)
{
    /* Calibration factor depends on sensor and lens */
    int position_change = (int)(phase_diff * calibration_factor);
    
    return current_position + position_change;
}
```

#### 2.2 Hybrid AF (PDAF + Contrast)

```c
/**
 * @brief Hybrid auto-focus combining PDAF and contrast AF
 */
struct hybrid_af_state {
    bool pdaf_available;
    float pdaf_confidence;
    
    int coarse_position;  /* From PDAF */
    int fine_position;    /* From contrast AF */
    
    enum {
        AF_IDLE,
        AF_PDAF_COARSE,
        AF_CONTRAST_FINE,
        AF_FOCUSED
    } state;
};

static int hybrid_af_step(
    struct hybrid_af_state *af,
    uint8_t *image,
    struct pdaf_pixel_pair *pdaf_pixels,
    unsigned int num_pdaf_pixels,
    int (*move_lens)(int position))
{
    switch (af->state) {
    case AF_IDLE:
        if (af->pdaf_available) {
            af->state = AF_PDAF_COARSE;
        } else {
            af->state = AF_CONTRAST_FINE;
        }
        break;
    
    case AF_PDAF_COARSE:
        /* Use PDAF for coarse positioning */
        float phase_diff = calculate_phase_difference(pdaf_pixels,
                                                      num_pdaf_pixels);
        
        af->coarse_position = pdaf_to_lens_position(phase_diff,
                                                    af->coarse_position,
                                                    1.5f);
        
        move_lens(af->coarse_position);
        
        /* Check PDAF confidence */
        if (af->pdaf_confidence > 0.8f) {
            /* High confidence, skip contrast AF */
            af->state = AF_FOCUSED;
        } else {
            /* Low confidence, refine with contrast AF */
            af->state = AF_CONTRAST_FINE;
        }
        break;
    
    case AF_CONTRAST_FINE:
        /* Fine-tune using contrast AF */
        /* Use small search range around PDAF position */
        /* ... contrast AF implementation ... */
        af->state = AF_FOCUSED;
        break;
    
    case AF_FOCUSED:
        return 1;  /* Focus complete */
    }
    
    return 0;
}
```

### 🔹 Part 3: Continuous Auto-Focus (CAF)

#### 3.1 CAF State Machine

```c
/**
 * @brief Continuous auto-focus for video
 */
struct caf_state {
    enum {
        CAF_STABLE,      /* In focus */
        CAF_UNSTABLE,    /* Focus drifting */
        CAF_SEARCHING,   /* Actively searching */
        CAF_TRACKING     /* Tracking moving subject */
    } state;
    
    float current_metric;
    float stable_metric;
    float metric_threshold;
    
    unsigned int stable_frames;
    unsigned int unstable_frames;
    
    int focus_position;
    int search_direction;
};

static void caf_update(
    struct caf_state *caf,
    float (*metric_func)(void*),
    void *metric_data,
    int (*move_lens)(int position))
{
    caf->current_metric = metric_func(metric_data);
    
    switch (caf->state) {
    case CAF_STABLE:
        /* Monitor for focus drift */
        if (fabsf(caf->current_metric - caf->stable_metric) > 
            caf->metric_threshold) {
            caf->unstable_frames++;
            
            if (caf->unstable_frames > 3) {
                /* Focus drifting, start search */
                caf->state = CAF_UNSTABLE;
                caf->unstable_frames = 0;
            }
        } else {
            caf->unstable_frames = 0;
            caf->stable_frames++;
        }
        break;
    
    case CAF_UNSTABLE:
        /* Determine search direction */
        int test_pos = caf->focus_position + 10;
        move_lens(test_pos);
        usleep(30000);
        
        float test_metric = metric_func(metric_data);
        
        if (test_metric > caf->current_metric) {
            caf->search_direction = 1;
        } else {
            caf->search_direction = -1;
        }
        
        move_lens(caf->focus_position);
        caf->state = CAF_SEARCHING;
        break;
    
    case CAF_SEARCHING:
        /* Search for peak */
        caf->focus_position += caf->search_direction * 5;
        move_lens(caf->focus_position);
        usleep(30000);
        
        float new_metric = metric_func(metric_data);
        
        if (new_metric < caf->current_metric) {
            /* Passed peak, go back */
            caf->focus_position -= caf->search_direction * 5;
            move_lens(caf->focus_position);
            
            caf->stable_metric = caf->current_metric;
            caf->state = CAF_STABLE;
            caf->stable_frames = 0;
        }
        
        caf->current_metric = new_metric;
        break;
    
    case CAF_TRACKING:
        /* Track moving subject */
        /* Predict motion and adjust focus proactively */
        break;
    }
}
```

#### 3.2 Scene Change Detection

```c
/**
 * @brief Detect scene changes to trigger refocus
 */
static bool detect_scene_change(
    uint8_t *prev_frame,
    uint8_t *curr_frame,
    unsigned int width,
    unsigned int height,
    float threshold)
{
    uint64_t diff_sum = 0;
    
    /* Sample pixels (not all for performance) */
    for (unsigned int y = 0; y < height; y += 4) {
        for (unsigned int x = 0; x < width; x += 4) {
            unsigned int idx = y * width + x;
            int diff = abs(curr_frame[idx] - prev_frame[idx]);
            diff_sum += diff;
        }
    }
    
    float avg_diff = (float)diff_sum / ((width/4) * (height/4));
    
    return avg_diff > threshold;
}
```

---

## 💻 Implementation Examples

### Example 1: Complete Auto-Focus System

```c
/**
 * @file autofocus_system.c
 * @brief Complete auto-focus implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

struct af_system {
    /* Hardware interface */
    int (*move_lens)(int position);
    int (*get_lens_position)(void);
    
    /* Configuration */
    int min_position;
    int max_position;
    
    struct {
        unsigned int x, y;
        unsigned int width, height;
    } focus_roi;
    
    /* Focus metric */
    enum {
        METRIC_VARIANCE,
        METRIC_LAPLACIAN,
        METRIC_TENENGRAD
    } metric_type;
    
    /* Search algorithm */
    enum {
        SEARCH_HILL_CLIMBING,
        SEARCH_BINARY,
        SEARCH_FIBONACCI
    } search_type;
    
    /* State */
    struct focus_search_state search_state;
    bool af_active;
    
    /* Statistics */
    unsigned int focus_time_ms;
    float final_metric;
};

/**
 * @brief Initialize AF system
 */
static struct af_system *af_system_init(
    int (*move_lens)(int),
    int (*get_lens)(void),
    int min_pos,
    int max_pos)
{
    struct af_system *af = calloc(1, sizeof(*af));
    
    af->move_lens = move_lens;
    af->get_lens_position = get_lens;
    af->min_position = min_pos;
    af->max_position = max_pos;
    
    /* Default ROI: center 50% */
    af->focus_roi.width = 640;
    af->focus_roi.height = 480;
    af->focus_roi.x = (1280 - 640) / 2;
    af->focus_roi.y = (720 - 480) / 2;
    
    /* Default metric and search */
    af->metric_type = METRIC_TENENGRAD;
    af->search_type = SEARCH_HILL_CLIMBING;
    
    return af;
}

/**
 * @brief Compute focus metric for current frame
 */
static float af_compute_metric(
    struct af_system *af,
    uint8_t *image,
    unsigned int width,
    unsigned int height)
{
    switch (af->metric_type) {
    case METRIC_VARIANCE:
        return focus_metric_variance(image, width, height,
                                     af->focus_roi.x, af->focus_roi.y,
                                     af->focus_roi.width, af->focus_roi.height);
    
    case METRIC_LAPLACIAN:
        return focus_metric_laplacian(image, width, height,
                                      af->focus_roi.x, af->focus_roi.y,
                                      af->focus_roi.width, af->focus_roi.height);
    
    case METRIC_TENENGRAD:
        return focus_metric_tenengrad(image, width, height,
                                      af->focus_roi.x, af->focus_roi.y,
                                      af->focus_roi.width, af->focus_roi.height);
    
    default:
        return 0.0f;
    }
}

/**
 * @brief Start auto-focus
 */
static void af_start(struct af_system *af)
{
    af->af_active = true;
    
    /* Initialize search state */
    af->search_state.current_position = af->get_lens_position();
    af->search_state.min_position = af->min_position;
    af->search_state.max_position = af->max_position;
    af->search_state.step_size = (af->max_position - af->min_position) / 10;
    af->search_state.state = SEARCH_COARSE;
    af->search_state.prev_metric = 0.0f;
    
    printf("Auto-focus started\n");
    printf("  Search range: %d - %d\n", af->min_position, af->max_position);
    printf("  Initial step: %d\n", af->search_state.step_size);
}

/**
 * @brief Update auto-focus (call for each frame)
 */
static bool af_update(
    struct af_system *af,
    uint8_t *image,
    unsigned int width,
    unsigned int height)
{
    if (!af->af_active)
        return true;
    
    /* Compute focus metric */
    float metric = af_compute_metric(af, image, width, height);
    
    /* Run search algorithm */
    bool done = false;
    
    switch (af->search_type) {
    case SEARCH_HILL_CLIMBING:
        done = focus_search_hill_climbing(&af->search_state,
                                          (void*)metric,
                                          NULL,
                                          af->move_lens);
        break;
    
    /* Other search algorithms... */
    }
    
    if (done) {
        af->af_active = false;
        af->final_metric = metric;
        
        printf("Auto-focus complete\n");
        printf("  Final position: %d\n", af->get_lens_position());
        printf("  Final metric: %.2f\n", af->final_metric);
        
        return true;
    }
    
    return false;
}

/**
 * @brief Example usage
 */
int main(void)
{
    /* Initialize lens control */
    /* ... hardware-specific initialization ... */
    
    /* Initialize AF system */
    struct af_system *af = af_system_init(
        lens_move,
        lens_get_position,
        0,    /* Min position */
        1023  /* Max position */
    );
    
    /* Open camera */
    /* ... V4L2 setup ... */
    
    /* Start auto-focus */
    af_start(af);
    
    /* Process frames */
    while (!af_update(af, frame_buffer, 1280, 720)) {
        /* Capture next frame */
        /* ... */
    }
    
    /* Cleanup */
    free(af);
    
    return 0;
}
```

### Example 2: Lens Driver with VCM Control

```c
/**
 * @file vcm_driver.c
 * @brief Voice Coil Motor driver for lens control
 */

#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/delay.h>

#define VCM_I2C_ADDR 0x0C

struct vcm_device {
    struct i2c_client *client;
    
    int current_position;
    int min_position;
    int max_position;
    
    /* Timing */
    unsigned int settle_time_us;
};

/**
 * @brief Move VCM to position
 */
static int vcm_move(struct vcm_device *vcm, int position)
{
    uint8_t data[2];
    
    /* Clamp position */
    if (position < vcm->min_position)
        position = vcm->min_position;
    if (position > vcm->max_position)
        position = vcm->max_position;
    
    /* VCM position format (10-bit) */
    data[0] = (position >> 4) & 0x3F;
    data[1] = (position << 4) & 0xF0;
    
    /* Write to VCM */
    if (i2c_master_send(vcm->client, data, 2) != 2) {
        dev_err(&vcm->client->dev, "Failed to move VCM\n");
        return -EIO;
    }
    
    vcm->current_position = position;
    
    /* Wait for lens to settle */
    usleep_range(vcm->settle_time_us, vcm->settle_time_us + 1000);
    
    return 0;
}

/**
 * @brief Get current VCM position
 */
static int vcm_get_position(struct vcm_device *vcm)
{
    return vcm->current_position;
}

/**
 * @brief Sysfs interface for manual control
 */
static ssize_t position_show(struct device *dev,
                            struct device_attribute *attr,
                            char *buf)
{
    struct vcm_device *vcm = dev_get_drvdata(dev);
    return sprintf(buf, "%d\n", vcm->current_position);
}

static ssize_t position_store(struct device *dev,
                             struct device_attribute *attr,
                             const char *buf,
                             size_t count)
{
    struct vcm_device *vcm = dev_get_drvdata(dev);
    int position;
    
    if (kstrtoint(buf, 10, &position))
        return -EINVAL;
    
    vcm_move(vcm, position);
    
    return count;
}

static DEVICE_ATTR_RW(position);

/**
 * @brief I2C probe
 */
static int vcm_probe(struct i2c_client *client,
                    const struct i2c_device_id *id)
{
    struct vcm_device *vcm;
    int ret;
    
    vcm = devm_kzalloc(&client->dev, sizeof(*vcm), GFP_KERNEL);
    if (!vcm)
        return -ENOMEM;
    
    vcm->client = client;
    vcm->min_position = 0;
    vcm->max_position = 1023;
    vcm->settle_time_us = 50000;  /* 50ms */
    
    i2c_set_clientdata(client, vcm);
    
    /* Create sysfs interface */
    ret = device_create_file(&client->dev, &dev_attr_position);
    if (ret) {
        dev_err(&client->dev, "Failed to create sysfs file\n");
        return ret;
    }
    
    /* Initialize to middle position */
    vcm_move(vcm, 512);
    
    dev_info(&client->dev, "VCM driver probed\n");
    
    return 0;
}

static const struct i2c_device_id vcm_id[] = {
    { "vcm-lens", 0 },
    { }
};
MODULE_DEVICE_TABLE(i2c, vcm_id);

static struct i2c_driver vcm_driver = {
    .driver = {
        .name = "vcm-lens",
    },
    .probe = vcm_probe,
    .id_table = vcm_id,
};

module_i2c_driver(vcm_driver);

MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("VCM Lens Driver");
MODULE_LICENSE("GPL v2");
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Focus Metric Comparison

```python
#!/usr/bin/env python3
"""
compare_focus_metrics.py
Compare different focus metrics
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def focus_variance(image):
    """Variance-based focus metric"""
    return np.var(image)

def focus_laplacian(image):
    """Laplacian-based focus metric"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.mean(np.abs(laplacian))

def focus_tenengrad(image):
    """Tenengrad (Sobel) focus metric"""
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gx**2 + gy**2)
    return np.sum(gradient[gradient > 10]**2)

def focus_fft(image):
    """FFT-based focus metric"""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # High frequency components
    h, w = image.shape
    center_h, center_w = h//2, w//2
    mask = np.zeros((h, w))
    
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            if dist > w/4:
                mask[y, x] = 1
    
    return np.sum(magnitude * mask)

# Test with focus sweep
cap = cv2.VideoCapture(0)

# Manually adjust focus or use motorized lens
focus_positions = range(0, 1024, 50)
metrics = {
    'Variance': [],
    'Laplacian': [],
    'Tenengrad': [],
    'FFT': []
}

print("Capturing focus sweep...")
for pos in focus_positions:
    # Set focus position (hardware-specific)
    # set_focus_position(pos)
    
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute metrics
    metrics['Variance'].append(focus_variance(gray))
    metrics['Laplacian'].append(focus_laplacian(gray))
    metrics['Tenengrad'].append(focus_tenengrad(gray))
    metrics['FFT'].append(focus_fft(gray))
    
    print(f"Position {pos}: done")

cap.release()

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (name, values) in enumerate(metrics.items()):
    ax = axes[idx // 2, idx % 2]
    ax.plot(focus_positions[:len(values)], values, 'o-')
    ax.set_xlabel('Focus Position')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{name} Focus Metric')
    ax.grid(True)
    
    # Mark peak
    peak_idx = np.argmax(values)
    peak_pos = focus_positions[peak_idx]
    ax.axvline(x=peak_pos, color='r', linestyle='--',
              label=f'Peak at {peak_pos}')
    ax.legend()

plt.tight_layout()
plt.savefig('focus_metrics_comparison.png')
plt.show()

print("\nPeak positions:")
for name, values in metrics.items():
    peak_idx = np.argmax(values)
    peak_pos = focus_positions[peak_idx]
    print(f"  {name}: {peak_pos}")
```

### Lab 2: AF Performance Measurement

```bash
#!/bin/bash
# measure_af_performance.sh

echo "Auto-Focus Performance Test"
echo "==========================="

# Test parameters
NUM_TESTS=10

# Results
declare -a focus_times
declare -a final_positions

for i in $(seq 1 $NUM_TESTS); do
    echo ""
    echo "Test $i/$NUM_TESTS"
    
    # Move to random start position
    start_pos=$((RANDOM % 1024))
    echo "  Start position: $start_pos"
    
    # Run AF
    start_time=$(date +%s%3N)
    ./autofocus_test $start_pos > af_log_$i.txt
    end_time=$(date +%s%3N)
    
    # Calculate time
    time_ms=$((end_time - start_pos))
    focus_times[$i]=$time_ms
    
    # Extract final position
    final_pos=$(grep "Final position" af_log_$i.txt | awk '{print $3}')
    final_positions[$i]=$final_pos
    
    echo "  Focus time: ${time_ms}ms"
    echo "  Final position: $final_pos"
done

# Calculate statistics
python3 << EOF
import numpy as np

times = [${focus_times[@]}]
positions = [${final_positions[@]}]

print("\nPerformance Statistics:")
print(f"  Mean focus time: {np.mean(times):.1f} ms")
print(f"  Std dev: {np.std(times):.1f} ms")
print(f"  Min: {np.min(times)} ms")
print(f"  Max: {np.max(times)} ms")

print(f"\nFinal Position Statistics:")
print(f"  Mean: {np.mean(positions):.1f}")
print(f"  Std dev: {np.std(positions):.1f}")
print(f"  Range: {np.min(positions)} - {np.max(positions)}")
EOF
```

---

## 🐛 Debugging Techniques

### Debug 1: Focus Hunting

**Symptoms:** Lens oscillates around focus point.

**Analysis:**

```c
/**
 * @brief Detect focus hunting
 */
struct hunting_detector {
    int positions[10];
    unsigned int count;
    unsigned int oscillations;
};

static bool detect_hunting(
    struct hunting_detector *detector,
    int current_position)
{
    if (detector->count < 10) {
        detector->positions[detector->count++] = current_position;
        return false;
    }
    
    /* Shift buffer */
    for (unsigned int i = 0; i < 9; i++) {
        detector->positions[i] = detector->positions[i+1];
    }
    detector->positions[9] = current_position;
    
    /* Count direction changes */
    unsigned int changes = 0;
    for (unsigned int i = 1; i < 9; i++) {
        int prev_dir = detector->positions[i] - detector->positions[i-1];
        int curr_dir = detector->positions[i+1] - detector->positions[i];
        
        if ((prev_dir > 0 && curr_dir < 0) ||
            (prev_dir < 0 && curr_dir > 0)) {
            changes++;
        }
    }
    
    /* Hunting if > 3 direction changes in 10 samples */
    return changes > 3;
}
```

**Fix:** Increase convergence threshold, add damping.

### Debug 2: Slow AF Speed

**Profiling:**

```c
/**
 * @brief Profile AF performance
 */
struct af_profile {
    uint64_t metric_time_us;
    uint64_t lens_move_time_us;
    uint64_t total_time_us;
    unsigned int num_steps;
};

static void af_profile_start(struct af_profile *prof)
{
    prof->metric_time_us = 0;
    prof->lens_move_time_us = 0;
    prof->total_time_us = 0;
    prof->num_steps = 0;
}

static void af_profile_end(struct af_profile *prof)
{
    printf("AF Performance Profile:\n");
    printf("  Total time: %llu ms\n", prof->total_time_us / 1000);
    printf("  Metric computation: %llu ms (%.1f%%)\n",
           prof->metric_time_us / 1000,
           100.0f * prof->metric_time_us / prof->total_time_us);
    printf("  Lens movement: %llu ms (%.1f%%)\n",
           prof->lens_move_time_us / 1000,
           100.0f * prof->lens_move_time_us / prof->total_time_us);
    printf("  Number of steps: %u\n", prof->num_steps);
    printf("  Time per step: %.1f ms\n",
           (float)prof->total_time_us / prof->num_steps / 1000);
}
```

---

## ⚡ Performance Optimization

### Optimization 1: ROI-Based Focus

```c
/**
 * @brief Adaptive ROI for faster focus metric computation
 */
static void optimize_focus_roi(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    unsigned int *roi_x,
    unsigned int *roi_y,
    unsigned int *roi_width,
    unsigned int *roi_height)
{
    /* Find region with highest edge density */
    unsigned int best_x = 0, best_y = 0;
    float max_edge_density = 0.0f;
    
    unsigned int step = 64;
    unsigned int test_size = 128;
    
    for (unsigned int y = 0; y < height - test_size; y += step) {
        for (unsigned int x = 0; x < width - test_size; x += step) {
            float edge_density = focus_metric_laplacian(
                image, width, height,
                x, y, test_size, test_size
            );
            
            if (edge_density > max_edge_density) {
                max_edge_density = edge_density;
                best_x = x;
                best_y = y;
            }
        }
    }
    
    *roi_x = best_x;
    *roi_y = best_y;
    *roi_width = test_size;
    *roi_height = test_size;
}
```

### Optimization 2: Predictive AF

```c
/**
 * @brief Predict focus position based on depth estimation
 */
static int predict_focus_position(
    float estimated_depth,
    struct lens_calibration *calib)
{
    /* Lens equation: 1/f = 1/u + 1/v */
    /* Where u = object distance, v = image distance */
    
    float focal_length = calib->focal_length_mm;
    float object_distance = estimated_depth * 1000;  /* m to mm */
    
    float image_distance = (focal_length * object_distance) /
                          (object_distance - focal_length);
    
    /* Convert to lens position */
    int position = (int)((image_distance - calib->min_image_dist) *
                        calib->position_per_mm);
    
    return position;
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the difference between contrast AF and phase detection AF.**

2. **Why does focus hunting occur? How can it be prevented?**

3. **Compare different focus metrics: which is best for low-light?**

4. **What is the advantage of hybrid AF (PDAF + contrast)?**

5. **Design an AF system for:**
   - Macro photography (close-up)
   - Sports photography (fast-moving subjects)

### Practical Challenges

1. **Implement face-detection-based AF ROI selection.**

2. **Debug an AF system that fails in low-contrast scenes.**

3. **Optimize AF to complete in < 500ms.**

4. **Design CAF algorithm for smooth video recording.**

---

## 📚 Further Reading & Resources

### Papers
- "Autofocus Survey" - Pertuz et al.
- "Phase Detection Autofocus" - Various manufacturers

### Standards
- Camera & Imaging Products Association (CIPA) AF standards

### Tools
- **OpenCV:** cv::Laplacian for focus metrics
- **libcamera:** Auto-focus algorithms
- **Android Camera2 API:** AF modes and controls

---

## 🎓 Summary

Today we covered:
- ✅ Focus metrics (variance, Laplacian, Tenengrad, FFT)
- ✅ Focus search algorithms (hill climbing, binary, Fibonacci)
- ✅ Phase detection auto-focus (PDAF)
- ✅ Hybrid AF combining PDAF and contrast
- ✅ Continuous auto-focus (CAF) for video
- ✅ VCM driver implementation
- ✅ Performance optimization and debugging

**Key Takeaways:**
1. Choice of focus metric affects speed and accuracy
2. Search algorithm determines AF speed
3. PDAF provides fast coarse positioning
4. CAF requires careful state management to avoid hunting
5. ROI selection significantly impacts performance

**Next:** Day 13 - Advanced AE/AWB Algorithms

---

**Day 12 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
