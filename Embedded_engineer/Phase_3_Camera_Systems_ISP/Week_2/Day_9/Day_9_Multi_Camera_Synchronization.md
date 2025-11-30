# Day 9: Multi-Camera Systems and Synchronization
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** multi-camera system architectures and use cases
2. **Implement** hardware and software camera synchronization
3. **Configure** frame synchronization and trigger mechanisms
4. **Develop** multi-camera capture and processing pipelines
5. **Debug** synchronization issues and timing problems
6. **Optimize** multi-camera bandwidth and performance

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Multiple camera modules, sync trigger hardware
*   **Software:** V4L2 multi-device support, synchronization libraries
*   **Knowledge:** Timing analysis, hardware triggers, DMA
*   **Tools:** Logic analyzer, oscilloscope for timing verification

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Multi-Camera Architectures

#### 1.1 Use Cases

**Automotive:**
- Surround view (4-6 cameras)
- Stereo vision (2 cameras)
- Driver monitoring + road view
- Mirror replacement

**Industrial:**
- 3D scanning (multiple angles)
- Quality inspection
- Motion capture
- Volumetric capture

**Consumer:**
- Dual camera phones (wide + telephoto)
- Depth sensing
- 360° video
- Multi-view streaming

#### 1.2 System Topologies

**Independent Cameras:**
```
Camera 1 → CSI-2 → ISP 1 → Memory
Camera 2 → CSI-2 → ISP 2 → Memory
Camera 3 → CSI-2 → ISP 3 → Memory
Camera 4 → CSI-2 → ISP 4 → Memory
```

**Shared ISP:**
```
Camera 1 ─┐
Camera 2 ─┼→ CSI-2 MUX → ISP → Memory
Camera 3 ─┤
Camera 4 ─┘
```

**Virtual Channels:**
```
Camera 1 → VC0 ─┐
Camera 2 → VC1 ─┼→ CSI-2 → ISP → Memory
Camera 3 → VC2 ─┤
Camera 4 → VC3 ─┘
```

#### 1.3 Synchronization Requirements

**Timing Accuracy:**

```c
/**
 * @brief Synchronization accuracy requirements
 */
struct sync_requirements {
    const char *application;
    unsigned int max_skew_us;  /* Maximum inter-camera skew */
    bool frame_sync_required;
    bool line_sync_required;
};

static const struct sync_requirements sync_reqs[] = {
    {"Surround View", 1000, true, false},      /* 1ms */
    {"Stereo Vision", 100, true, false},       /* 100us */
    {"3D Scanning", 10, true, true},           /* 10us */
    {"High-Speed Sync", 1, true, true},        /* 1us */
};
```

### 🔹 Part 2: Hardware Synchronization

#### 2.1 External Trigger

**Master-Slave Configuration:**

```
Master Camera:
  - Generates VSYNC/HSYNC
  - Outputs trigger signal
  
Slave Cameras:
  - Receive trigger
  - Synchronize frame start
```

**Hardware Connections:**

```
Master Camera GPIO → Slave 1 Trigger
                  → Slave 2 Trigger
                  → Slave 3 Trigger
```

**Implementation:**

```c
/**
 * @file camera_sync_hw.c
 * @brief Hardware-based camera synchronization
 */

#include <linux/gpio.h>
#include <linux/interrupt.h>

struct camera_sync_hw {
    struct gpio_desc *trigger_out;  /* Master trigger output */
    struct gpio_desc *trigger_in;   /* Slave trigger input */
    int trigger_irq;
    
    /* Timing */
    ktime_t last_trigger;
    unsigned int frame_count;
    
    /* Callback */
    void (*trigger_callback)(void *data);
    void *callback_data;
};

/**
 * @brief Configure master camera for trigger output
 */
static int camera_sync_master_init(
    struct camera_sync_hw *sync,
    struct device *dev)
{
    /* Get GPIO for trigger output */
    sync->trigger_out = devm_gpiod_get(dev, "trigger", GPIOD_OUT_LOW);
    if (IS_ERR(sync->trigger_out))
        return PTR_ERR(sync->trigger_out);
    
    /* Configure sensor to output VSYNC on GPIO */
    /* Sensor-specific register configuration */
    sensor_write_reg(dev, 0x3040, 0x01);  /* Enable VSYNC output */
    sensor_write_reg(dev, 0x3041, 0x00);  /* Active high */
    
    dev_info(dev, "Master camera sync initialized\n");
    
    return 0;
}

/**
 * @brief Trigger interrupt handler for slave camera
 */
static irqreturn_t trigger_irq_handler(int irq, void *dev_id)
{
    struct camera_sync_hw *sync = dev_id;
    ktime_t now = ktime_get();
    
    /* Record timing */
    if (sync->frame_count > 0) {
        s64 interval_us = ktime_us_delta(now, sync->last_trigger);
        pr_debug("Frame interval: %lld us\n", interval_us);
    }
    
    sync->last_trigger = now;
    sync->frame_count++;
    
    /* Trigger callback */
    if (sync->trigger_callback) {
        sync->trigger_callback(sync->callback_data);
    }
    
    return IRQ_HANDLED;
}

/**
 * @brief Configure slave camera for trigger input
 */
static int camera_sync_slave_init(
    struct camera_sync_hw *sync,
    struct device *dev)
{
    int ret;
    
    /* Get GPIO for trigger input */
    sync->trigger_in = devm_gpiod_get(dev, "trigger", GPIOD_IN);
    if (IS_ERR(sync->trigger_in))
        return PTR_ERR(sync->trigger_in);
    
    /* Get IRQ */
    sync->trigger_irq = gpiod_to_irq(sync->trigger_in);
    if (sync->trigger_irq < 0)
        return sync->trigger_irq;
    
    /* Request IRQ */
    ret = devm_request_irq(dev, sync->trigger_irq,
                          trigger_irq_handler,
                          IRQF_TRIGGER_RISING,
                          "camera-trigger", sync);
    if (ret) {
        dev_err(dev, "Failed to request trigger IRQ\n");
        return ret;
    }
    
    /* Configure sensor for external trigger */
    sensor_write_reg(dev, 0x3042, 0x01);  /* Enable external trigger */
    sensor_write_reg(dev, 0x3043, 0x00);  /* Trigger on rising edge */
    
    dev_info(dev, "Slave camera sync initialized\n");
    
    return 0;
}
```

#### 2.2 Genlock (Frame Synchronization)

**Concept:**
- All cameras share common timing reference
- Frame starts aligned
- Pixel clock may be independent

```c
/**
 * @brief Genlock configuration
 */
struct genlock_config {
    unsigned int frame_rate;
    unsigned int h_total;
    unsigned int v_total;
    bool master;
};

static int configure_genlock(
    struct camera_device *cam,
    struct genlock_config *config)
{
    if (config->master) {
        /* Master: generate timing */
        sensor_write_reg(cam, REG_GENLOCK_MODE, 0x01);  /* Master */
        sensor_write_reg(cam, REG_FRAME_RATE, config->frame_rate);
    } else {
        /* Slave: lock to external timing */
        sensor_write_reg(cam, REG_GENLOCK_MODE, 0x02);  /* Slave */
        
        /* Configure PLL to lock to external reference */
        sensor_write_reg(cam, REG_PLL_CTRL, 0x80);  /* Enable external ref */
    }
    
    return 0;
}
```

### 🔹 Part 3: Software Synchronization

#### 3.1 Timestamp-Based Sync

**V4L2 Timestamps:**

```c
/**
 * @brief Multi-camera capture with timestamp synchronization
 */

struct multi_camera_capture {
    unsigned int num_cameras;
    int *fds;  /* V4L2 device file descriptors */
    
    struct {
        void *buffer;
        size_t size;
        struct timeval timestamp;
    } *frames;
    
    uint64_t max_skew_us;
};

/**
 * @brief Capture synchronized frame set
 */
static int capture_synchronized_frames(
    struct multi_camera_capture *mcc)
{
    struct timeval reference_time = {0};
    bool have_reference = false;
    
    /* Capture from all cameras */
    for (unsigned int i = 0; i < mcc->num_cameras; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        /* Dequeue buffer */
        if (ioctl(mcc->fds[i], VIDIOC_DQBUF, &buf) < 0) {
            perror("VIDIOC_DQBUF");
            return -1;
        }
        
        /* Store frame and timestamp */
        mcc->frames[i].timestamp = buf.timestamp;
        
        /* Set reference time from first camera */
        if (!have_reference) {
            reference_time = buf.timestamp;
            have_reference = true;
        }
        
        /* Check synchronization */
        uint64_t ref_us = reference_time.tv_sec * 1000000ULL + 
                         reference_time.tv_usec;
        uint64_t cam_us = buf.timestamp.tv_sec * 1000000ULL + 
                         buf.timestamp.tv_usec;
        uint64_t skew = (cam_us > ref_us) ? (cam_us - ref_us) : 
                                            (ref_us - cam_us);
        
        if (skew > mcc->max_skew_us) {
            pr_warn("Camera %u: skew %llu us exceeds limit %llu us\n",
                   i, skew, mcc->max_skew_us);
        }
        
        /* Requeue buffer */
        if (ioctl(mcc->fds[i], VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            return -1;
        }
    }
    
    return 0;
}
```

#### 3.2 PTP (Precision Time Protocol)

**IEEE 1588 for Camera Sync:**

```c
/**
 * @brief PTP-based camera synchronization
 */

#include <linux/ptp_clock_kernel.h>

struct camera_ptp {
    struct ptp_clock_info ptp_info;
    struct ptp_clock *ptp_clock;
    
    /* Hardware timestamp counter */
    void __iomem *timestamp_reg;
    unsigned int timestamp_freq;
};

/**
 * @brief Get hardware timestamp
 */
static int camera_ptp_gettime(
    struct ptp_clock_info *ptp,
    struct timespec64 *ts)
{
    struct camera_ptp *cam_ptp = container_of(ptp, struct camera_ptp, ptp_info);
    
    /* Read hardware timestamp counter */
    uint64_t hw_timestamp = readq(cam_ptp->timestamp_reg);
    
    /* Convert to timespec */
    uint64_t ns = (hw_timestamp * 1000000000ULL) / cam_ptp->timestamp_freq;
    
    ts->tv_sec = ns / 1000000000ULL;
    ts->tv_nsec = ns % 1000000000ULL;
    
    return 0;
}

/**
 * @brief Set hardware timestamp
 */
static int camera_ptp_settime(
    struct ptp_clock_info *ptp,
    const struct timespec64 *ts)
{
    struct camera_ptp *cam_ptp = container_of(ptp, struct camera_ptp, ptp_info);
    
    /* Convert timespec to hardware counter value */
    uint64_t ns = ts->tv_sec * 1000000000ULL + ts->tv_nsec;
    uint64_t hw_timestamp = (ns * cam_ptp->timestamp_freq) / 1000000000ULL;
    
    /* Write to hardware */
    writeq(hw_timestamp, cam_ptp->timestamp_reg);
    
    return 0;
}

static struct ptp_clock_info camera_ptp_info = {
    .owner = THIS_MODULE,
    .name = "camera_ptp",
    .max_adj = 500000,
    .n_alarm = 0,
    .n_ext_ts = 0,
    .n_per_out = 0,
    .pps = 0,
    .gettime64 = camera_ptp_gettime,
    .settime64 = camera_ptp_settime,
};
```

### 🔹 Part 4: Multi-Camera Processing

#### 4.1 Parallel Capture

**Thread-Per-Camera:**

```c
/**
 * @brief Multi-camera parallel capture
 */

struct camera_thread_data {
    int camera_id;
    int fd;
    struct camera_buffer *buffers;
    unsigned int num_buffers;
    
    /* Synchronization */
    pthread_barrier_t *frame_barrier;
    
    /* Output */
    void (*frame_callback)(int camera_id, void *data, size_t size);
};

static void *camera_capture_thread(void *arg)
{
    struct camera_thread_data *data = arg;
    
    while (running) {
        /* Dequeue buffer */
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(data->fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("VIDIOC_DQBUF");
            break;
        }
        
        /* Wait for all cameras to capture */
        pthread_barrier_wait(data->frame_barrier);
        
        /* Process frame */
        if (data->frame_callback) {
            data->frame_callback(data->camera_id,
                                data->buffers[buf.index].start,
                                buf.bytesused);
        }
        
        /* Requeue */
        if (ioctl(data->fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            break;
        }
    }
    
    return NULL;
}

/**
 * @brief Start multi-camera capture
 */
static int start_multi_camera_capture(
    struct multi_camera_system *mcs)
{
    pthread_barrier_t frame_barrier;
    pthread_barrier_init(&frame_barrier, NULL, mcs->num_cameras);
    
    pthread_t threads[mcs->num_cameras];
    struct camera_thread_data thread_data[mcs->num_cameras];
    
    for (unsigned int i = 0; i < mcs->num_cameras; i++) {
        thread_data[i].camera_id = i;
        thread_data[i].fd = mcs->cameras[i].fd;
        thread_data[i].buffers = mcs->cameras[i].buffers;
        thread_data[i].num_buffers = mcs->cameras[i].num_buffers;
        thread_data[i].frame_barrier = &frame_barrier;
        thread_data[i].frame_callback = mcs->frame_callback;
        
        pthread_create(&threads[i], NULL, camera_capture_thread,
                      &thread_data[i]);
    }
    
    /* Wait for threads */
    for (unsigned int i = 0; i < mcs->num_cameras; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_barrier_destroy(&frame_barrier);
    
    return 0;
}
```

#### 4.2 Frame Alignment

**Temporal Alignment:**

```c
/**
 * @brief Align frames from multiple cameras temporally
 */

struct frame_queue {
    struct {
        void *data;
        uint64_t timestamp_us;
        bool valid;
    } frames[16];  /* Ring buffer */
    
    unsigned int head;
    unsigned int tail;
    unsigned int count;
};

struct multi_camera_aligner {
    unsigned int num_cameras;
    struct frame_queue *queues;  /* One per camera */
    
    uint64_t max_skew_us;
};

/**
 * @brief Add frame to queue
 */
static void frame_queue_push(
    struct frame_queue *q,
    void *data,
    uint64_t timestamp_us)
{
    q->frames[q->head].data = data;
    q->frames[q->head].timestamp_us = timestamp_us;
    q->frames[q->head].valid = true;
    
    q->head = (q->head + 1) % 16;
    if (q->count < 16)
        q->count++;
}

/**
 * @brief Get aligned frame set
 */
static bool get_aligned_frames(
    struct multi_camera_aligner *aligner,
    void **frames_out,
    uint64_t *timestamp_out)
{
    /* Find oldest frame across all cameras */
    uint64_t oldest_time = UINT64_MAX;
    
    for (unsigned int i = 0; i < aligner->num_cameras; i++) {
        struct frame_queue *q = &aligner->queues[i];
        
        if (q->count == 0)
            return false;  /* Not all cameras have frames */
        
        uint64_t cam_time = q->frames[q->tail].timestamp_us;
        if (cam_time < oldest_time)
            oldest_time = cam_time;
    }
    
    /* Check if all cameras have frames within skew tolerance */
    for (unsigned int i = 0; i < aligner->num_cameras; i++) {
        struct frame_queue *q = &aligner->queues[i];
        uint64_t cam_time = q->frames[q->tail].timestamp_us;
        
        if (cam_time - oldest_time > aligner->max_skew_us) {
            /* This camera is too far ahead, wait for others */
            return false;
        }
    }
    
    /* All frames are aligned, extract them */
    for (unsigned int i = 0; i < aligner->num_cameras; i++) {
        struct frame_queue *q = &aligner->queues[i];
        frames_out[i] = q->frames[q->tail].data;
        
        q->tail = (q->tail + 1) % 16;
        q->count--;
    }
    
    *timestamp_out = oldest_time;
    
    return true;
}
```

---

## 💻 Implementation Examples

### Example 1: Stereo Camera System

```c
/**
 * @file stereo_camera.c
 * @brief Synchronized stereo camera capture
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

struct stereo_camera {
    /* Left camera */
    int fd_left;
    struct camera_buffer buffers_left[4];
    
    /* Right camera */
    int fd_right;
    struct camera_buffer buffers_right[4];
    
    /* Configuration */
    unsigned int width;
    unsigned int height;
    
    /* Synchronization */
    pthread_mutex_t sync_mutex;
    pthread_cond_t frame_ready;
    
    struct {
        void *left;
        void *right;
        struct timeval timestamp;
        bool ready;
    } stereo_pair;
    
    /* Callback */
    void (*stereo_callback)(void *left, void *right, void *user_data);
    void *user_data;
};

/**
 * @brief Initialize stereo camera system
 */
static struct stereo_camera *stereo_camera_init(
    const char *dev_left,
    const char *dev_right,
    unsigned int width,
    unsigned int height)
{
    struct stereo_camera *stereo = calloc(1, sizeof(*stereo));
    
    stereo->width = width;
    stereo->height = height;
    
    /* Initialize left camera */
    stereo->fd_left = open(dev_left, O_RDWR);
    /* ... V4L2 setup ... */
    
    /* Initialize right camera */
    stereo->fd_right = open(dev_right, O_RDWR);
    /* ... V4L2 setup ... */
    
    /* Initialize synchronization */
    pthread_mutex_init(&stereo->sync_mutex, NULL);
    pthread_cond_init(&stereo->frame_ready, NULL);
    
    return stereo;
}

/**
 * @brief Left camera capture thread
 */
static void *left_camera_thread(void *arg)
{
    struct stereo_camera *stereo = arg;
    
    while (running) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(stereo->fd_left, VIDIOC_DQBUF, &buf) < 0)
            break;
        
        pthread_mutex_lock(&stereo->sync_mutex);
        
        stereo->stereo_pair.left = stereo->buffers_left[buf.index].start;
        stereo->stereo_pair.timestamp = buf.timestamp;
        
        /* Check if right frame is ready */
        if (stereo->stereo_pair.right != NULL) {
            stereo->stereo_pair.ready = true;
            pthread_cond_signal(&stereo->frame_ready);
        }
        
        pthread_mutex_unlock(&stereo->sync_mutex);
        
        /* Requeue */
        ioctl(stereo->fd_left, VIDIOC_QBUF, &buf);
    }
    
    return NULL;
}

/**
 * @brief Right camera capture thread
 */
static void *right_camera_thread(void *arg)
{
    struct stereo_camera *stereo = arg;
    
    while (running) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(stereo->fd_right, VIDIOC_DQBUF, &buf) < 0)
            break;
        
        pthread_mutex_lock(&stereo->sync_mutex);
        
        stereo->stereo_pair.right = stereo->buffers_right[buf.index].start;
        
        /* Check if left frame is ready */
        if (stereo->stereo_pair.left != NULL) {
            stereo->stereo_pair.ready = true;
            pthread_cond_signal(&stereo->frame_ready);
        }
        
        pthread_mutex_unlock(&stereo->sync_mutex);
        
        /* Requeue */
        ioctl(stereo->fd_right, VIDIOC_QBUF, &buf);
    }
    
    return NULL;
}

/**
 * @brief Processing thread
 */
static void *stereo_processing_thread(void *arg)
{
    struct stereo_camera *stereo = arg;
    
    while (running) {
        pthread_mutex_lock(&stereo->sync_mutex);
        
        /* Wait for stereo pair */
        while (!stereo->stereo_pair.ready) {
            pthread_cond_wait(&stereo->frame_ready, &stereo->sync_mutex);
        }
        
        /* Process stereo pair */
        if (stereo->stereo_callback) {
            stereo->stereo_callback(stereo->stereo_pair.left,
                                   stereo->stereo_pair.right,
                                   stereo->user_data);
        }
        
        /* Reset */
        stereo->stereo_pair.left = NULL;
        stereo->stereo_pair.right = NULL;
        stereo->stereo_pair.ready = false;
        
        pthread_mutex_unlock(&stereo->sync_mutex);
    }
    
    return NULL;
}

/**
 * @brief Start stereo capture
 */
static int stereo_camera_start(struct stereo_camera *stereo)
{
    pthread_t thread_left, thread_right, thread_process;
    
    /* Start capture threads */
    pthread_create(&thread_left, NULL, left_camera_thread, stereo);
    pthread_create(&thread_right, NULL, right_camera_thread, stereo);
    pthread_create(&thread_process, NULL, stereo_processing_thread, stereo);
    
    /* Start streaming */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(stereo->fd_left, VIDIOC_STREAMON, &type);
    ioctl(stereo->fd_right, VIDIOC_STREAMON, &type);
    
    /* Wait */
    pthread_join(thread_left, NULL);
    pthread_join(thread_right, NULL);
    pthread_join(thread_process, NULL);
    
    return 0;
}
```

### Example 2: Surround View System (4 Cameras)

```c
/**
 * @file surround_view.c
 * @brief 4-camera surround view system
 */

#define NUM_CAMERAS 4

enum camera_position {
    CAM_FRONT = 0,
    CAM_REAR = 1,
    CAM_LEFT = 2,
    CAM_RIGHT = 3
};

struct surround_view_system {
    struct {
        int fd;
        struct camera_buffer buffers[4];
        const char *name;
    } cameras[NUM_CAMERAS];
    
    /* Calibration */
    struct {
        float intrinsic[3][3];
        float distortion[5];
        float extrinsic[4][4];
    } calibration[NUM_CAMERAS];
    
    /* Output */
    uint8_t *bird_eye_view;
    unsigned int bev_width;
    unsigned int bev_height;
};

/**
 * @brief Initialize surround view system
 */
static struct surround_view_system *surround_view_init(void)
{
    struct surround_view_system *sv = calloc(1, sizeof(*sv));
    
    /* Initialize cameras */
    sv->cameras[CAM_FRONT].name = "Front";
    sv->cameras[CAM_REAR].name = "Rear";
    sv->cameras[CAM_LEFT].name = "Left";
    sv->cameras[CAM_RIGHT].name = "Right";
    
    for (int i = 0; i < NUM_CAMERAS; i++) {
        char dev_name[32];
        snprintf(dev_name, sizeof(dev_name), "/dev/video%d", i);
        
        sv->cameras[i].fd = open(dev_name, O_RDWR);
        /* ... V4L2 setup ... */
        
        printf("Initialized %s camera\n", sv->cameras[i].name);
    }
    
    /* Allocate bird's eye view buffer */
    sv->bev_width = 1024;
    sv->bev_height = 1024;
    sv->bev_view = malloc(sv->bev_width * sv->bev_height * 3);
    
    return sv;
}

/**
 * @brief Capture synchronized frames from all cameras
 */
static int surround_view_capture(
    struct surround_view_system *sv,
    void **frames_out)
{
    struct v4l2_buffer bufs[NUM_CAMERAS];
    
    /* Dequeue from all cameras */
    for (int i = 0; i < NUM_CAMERAS; i++) {
        memset(&bufs[i], 0, sizeof(bufs[i]));
        bufs[i].type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        bufs[i].memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(sv->cameras[i].fd, VIDIOC_DQBUF, &bufs[i]) < 0) {
            perror("VIDIOC_DQBUF");
            return -1;
        }
        
        frames_out[i] = sv->cameras[i].buffers[bufs[i].index].start;
    }
    
    /* Check synchronization */
    struct timeval ref_time = bufs[0].timestamp;
    for (int i = 1; i < NUM_CAMERAS; i++) {
        uint64_t ref_us = ref_time.tv_sec * 1000000ULL + ref_time.tv_usec;
        uint64_t cam_us = bufs[i].timestamp.tv_sec * 1000000ULL + 
                         bufs[i].timestamp.tv_usec;
        int64_t skew = (int64_t)cam_us - (int64_t)ref_us;
        
        if (abs(skew) > 1000) {  /* 1ms tolerance */
            pr_warn("%s camera: skew %lld us\n",
                   sv->cameras[i].name, skew);
        }
    }
    
    /* Requeue all buffers */
    for (int i = 0; i < NUM_CAMERAS; i++) {
        ioctl(sv->cameras[i].fd, VIDIOC_QBUF, &bufs[i]);
    }
    
    return 0;
}

/**
 * @brief Generate bird's eye view from 4 cameras
 */
static void generate_bird_eye_view(
    struct surround_view_system *sv,
    void **camera_frames)
{
    /* This would involve:
     * 1. Undistort each camera image
     * 2. Apply homography transformation
     * 3. Blend overlapping regions
     * 4. Composite into single bird's eye view
     */
    
    /* Simplified example - just copy center regions */
    for (int i = 0; i < NUM_CAMERAS; i++) {
        /* Extract and transform region from each camera */
        /* ... image processing ... */
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Measure Synchronization Accuracy

```bash
#!/bin/bash
# measure_sync_accuracy.sh

echo "Multi-Camera Sync Accuracy Test"
echo "================================"

# Capture frames with timestamps
for cam in 0 1; do
    v4l2-ctl --device=/dev/video${cam} \
             --stream-mmap \
             --stream-to=cam${cam}_frames.raw \
             --stream-count=100 \
             --verbose 2>&1 | grep "timestamp" > cam${cam}_timestamps.txt &
done

wait

# Analyze timestamps
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

def parse_timestamps(filename):
    timestamps = []
    with open(filename, 'r') as f:
        for line in f:
            # Extract timestamp from V4L2 output
            # Format: "timestamp: 1234567.123456"
            if 'timestamp' in line:
                ts_str = line.split(':')[1].strip()
                timestamps.append(float(ts_str))
    return np.array(timestamps)

ts0 = parse_timestamps('cam0_timestamps.txt')
ts1 = parse_timestamps('cam1_timestamps.txt')

# Calculate skew
min_len = min(len(ts0), len(ts1))
skew = (ts1[:min_len] - ts0[:min_len]) * 1000000  # Convert to microseconds

print(f"Synchronization Statistics:")
print(f"  Mean skew: {np.mean(skew):.1f} us")
print(f"  Std dev: {np.std(skew):.1f} us")
print(f"  Max skew: {np.max(np.abs(skew)):.1f} us")

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(skew)
plt.ylabel('Skew (us)')
plt.title('Inter-Camera Skew Over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.hist(skew, bins=50)
plt.xlabel('Skew (us)')
plt.ylabel('Count')
plt.title('Skew Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('sync_accuracy.png')
plt.show()
EOF
```

### Lab 2: Trigger Timing Verification

```c
/**
 * @brief Verify hardware trigger timing with GPIO
 */
#include <linux/gpio.h>
#include <linux/ktime.h>

struct trigger_timing_test {
    struct gpio_desc *trigger_gpio;
    struct gpio_desc *capture_gpio;
    
    ktime_t trigger_times[1000];
    ktime_t capture_times[1000];
    unsigned int count;
};

static irqreturn_t capture_irq_handler(int irq, void *dev_id)
{
    struct trigger_timing_test *test = dev_id;
    
    if (test->count < 1000) {
        test->capture_times[test->count] = ktime_get();
        test->count++;
    }
    
    return IRQ_HANDLED;
}

static void run_trigger_timing_test(struct trigger_timing_test *test)
{
    test->count = 0;
    
    /* Generate 1000 triggers */
    for (int i = 0; i < 1000; i++) {
        test->trigger_times[i] = ktime_get();
        gpiod_set_value(test->trigger_gpio, 1);
        udelay(10);
        gpiod_set_value(test->trigger_gpio, 0);
        msleep(10);
    }
    
    /* Analyze timing */
    for (int i = 0; i < test->count; i++) {
        s64 latency_ns = ktime_to_ns(ktime_sub(test->capture_times[i],
                                               test->trigger_times[i]));
        pr_info("Trigger %d: latency %lld ns\n", i, latency_ns);
    }
}
```

---

## 🐛 Debugging Techniques

### Debug 1: Frame Drop Detection

```c
/**
 * @brief Detect dropped frames in multi-camera system
 */
struct frame_drop_detector {
    unsigned int expected_frame_count[MAX_CAMERAS];
    unsigned int actual_frame_count[MAX_CAMERAS];
    unsigned int drops[MAX_CAMERAS];
};

static void check_frame_drops(
    struct frame_drop_detector *detector,
    unsigned int camera_id,
    unsigned int sequence_number)
{
    unsigned int expected = detector->expected_frame_count[camera_id];
    
    if (sequence_number != expected) {
        unsigned int dropped = sequence_number - expected;
        detector->drops[camera_id] += dropped;
        
        pr_warn("Camera %u: dropped %u frames (seq %u, expected %u)\n",
               camera_id, dropped, sequence_number, expected);
    }
    
    detector->expected_frame_count[camera_id] = sequence_number + 1;
    detector->actual_frame_count[camera_id]++;
}
```

### Debug 2: Bandwidth Analysis

```python
#!/usr/bin/env python3
"""
analyze_multi_camera_bandwidth.py
Analyze memory bandwidth for multi-camera system
"""

def calculate_bandwidth(num_cameras, width, height, fps, bpp):
    """Calculate total system bandwidth"""
    
    # Per-camera bandwidth
    pixels_per_sec = width * height * fps
    bits_per_sec = pixels_per_sec * bpp
    bytes_per_sec = bits_per_sec / 8
    
    # Total for all cameras
    total_bandwidth = bytes_per_sec * num_cameras
    
    # Add ISP processing overhead (~50%)
    total_bandwidth *= 1.5
    
    return total_bandwidth

# Test different configurations
configs = [
    (4, 1920, 1080, 30, 10, "4x 1080p@30 RAW10"),
    (6, 1280, 720, 30, 10, "6x 720p@30 RAW10"),
    (4, 1920, 1080, 60, 10, "4x 1080p@60 RAW10"),
]

print("Multi-Camera Bandwidth Analysis")
print("=" * 60)

for num_cam, w, h, fps, bpp, desc in configs:
    bw = calculate_bandwidth(num_cam, w, h, fps, bpp)
    bw_gbps = bw * 8 / 1e9
    
    print(f"\n{desc}:")
    print(f"  Total bandwidth: {bw/1e6:.1f} MB/s ({bw_gbps:.2f} Gbps)")
    print(f"  Per camera: {bw/num_cam/1e6:.1f} MB/s")
```

---

## ⚡ Performance Optimization

### Optimization 1: Zero-Copy Multi-Camera

```c
/**
 * @brief Zero-copy multi-camera pipeline using DMA-BUF
 */
static int setup_zero_copy_multi_camera(
    struct multi_camera_system *mcs)
{
    for (unsigned int i = 0; i < mcs->num_cameras; i++) {
        /* Export camera buffers as DMA-BUF */
        for (unsigned int j = 0; j < NUM_BUFFERS; j++) {
            struct v4l2_exportbuffer exp = {0};
            exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            exp.index = j;
            exp.flags = O_RDWR;
            
            if (ioctl(mcs->cameras[i].fd, VIDIOC_EXPBUF, &exp) < 0) {
                perror("VIDIOC_EXPBUF");
                return -1;
            }
            
            mcs->cameras[i].dmabuf_fds[j] = exp.fd;
        }
    }
    
    return 0;
}
```

### Optimization 2: Parallel ISP Processing

```c
/**
 * @brief Process multiple camera streams in parallel
 */
static void process_multi_camera_parallel(
    struct multi_camera_system *mcs,
    void **frames)
{
    #pragma omp parallel for
    for (unsigned int i = 0; i < mcs->num_cameras; i++) {
        /* Each camera processed on separate thread */
        isp_process(frames[i], mcs->cameras[i].output,
                   mcs->width, mcs->height);
    }
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the difference between hardware and software camera synchronization. When would you use each?**

2. **What causes frame skew in multi-camera systems? How can it be minimized?**

3. **Calculate total system bandwidth for:**
   - 6 cameras, 1080p@30fps, RAW10
   - Include ISP processing overhead

4. **Why is PTP (IEEE 1588) useful for multi-camera synchronization?**

5. **Design a synchronization strategy for:**
   - Automotive surround view (4 cameras)
   - High-speed 3D scanning (8 cameras)

### Practical Challenges

1. **Implement frame alignment algorithm for cameras with different frame rates.**

2. **Debug a system where cameras drift out of sync over time.**

3. **Optimize multi-camera capture to minimize latency.**

4. **Design hardware trigger circuit for 8-camera sync.**

---

## 📚 Further Reading & Resources

### Standards
- [IEEE 1588 (PTP)](https://standards.ieee.org/standard/1588-2019.html)
- [MIPI Camera Serial Interface](https://www.mipi.org/specifications/csi-2)

### Papers
- "Multi-Camera Synchronization" - Various authors
- "Surround View Systems for Automotive" - Industry papers

### Tools
- **ptp4l:** Linux PTP daemon
- **v4l2-ctl:** Multi-device testing
- **GStreamer:** Multi-camera pipelines

---

## 🎓 Summary

Today we covered:
- ✅ Multi-camera system architectures and use cases
- ✅ Hardware synchronization (triggers, genlock)
- ✅ Software synchronization (timestamps, PTP)
- ✅ Multi-camera capture and processing pipelines
- ✅ Frame alignment and temporal synchronization
- ✅ Stereo and surround view implementations
- ✅ Synchronization accuracy measurement and debugging

**Key Takeaways:**
1. Hardware sync provides best accuracy but requires additional connections
2. Software sync is flexible but limited by system latency
3. Frame alignment is critical for multi-camera applications
4. Bandwidth management is crucial for multi-camera systems

**Next:** Day 10 - Stereo Vision and Depth Estimation

---

**Day 9 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
