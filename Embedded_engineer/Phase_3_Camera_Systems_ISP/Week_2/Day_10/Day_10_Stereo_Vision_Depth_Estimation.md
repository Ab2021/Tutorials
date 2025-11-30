# Day 10: Stereo Vision and Depth Estimation
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** stereo vision principles and epipolar geometry
2. **Implement** stereo calibration and rectification algorithms
3. **Configure** stereo matching and disparity computation
4. **Develop** depth map generation and point cloud creation
5. **Debug** stereo vision artifacts and calibration errors
6. **Optimize** real-time stereo depth estimation performance

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Stereo camera rig with known baseline
*   **Software:** OpenCV, PCL (Point Cloud Library), calibration tools
*   **Knowledge:** Linear algebra, camera geometry, image processing
*   **Tools:** Calibration patterns (checkerboard), 3D visualization

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Stereo Vision Fundamentals

#### 1.1 Epipolar Geometry

**Basic Stereo Setup:**

```
Left Camera (CL)          Right Camera (CR)
      |                         |
      |<------- Baseline ------>|
      |                         |
      ↓                         ↓
   Image L                  Image R
   
Point P in 3D space projects to:
- pL in left image
- pR in right image
```

**Epipolar Constraint:**

```c
/**
 * @brief Epipolar geometry parameters
 */
struct epipolar_geometry {
    float baseline;        /* Distance between cameras (mm) */
    float focal_length;    /* Focal length (pixels) */
    
    /* Essential matrix E */
    float E[3][3];
    
    /* Fundamental matrix F */
    float F[3][3];
    
    /* Rotation and translation */
    float R[3][3];         /* Rotation from left to right */
    float t[3];            /* Translation vector */
};

/**
 * @brief Depth from disparity
 * 
 * Z = (f × B) / d
 * 
 * Where:
 * - Z: depth
 * - f: focal length
 * - B: baseline
 * - d: disparity (xL - xR)
 */
static float depth_from_disparity(
    float disparity,
    float focal_length,
    float baseline)
{
    if (disparity <= 0.0f)
        return INFINITY;
    
    return (focal_length * baseline) / disparity;
}

/**
 * @brief Calculate disparity range
 */
static void calculate_disparity_range(
    float min_depth,
    float max_depth,
    float focal_length,
    float baseline,
    int *min_disparity,
    int *max_disparity)
{
    /* Max disparity at min depth */
    *max_disparity = (int)((focal_length * baseline) / min_depth);
    
    /* Min disparity at max depth */
    *min_disparity = (int)((focal_length * baseline) / max_depth);
    
    /* Ensure minimum range */
    if (*min_disparity < 0)
        *min_disparity = 0;
}
```

#### 1.2 Stereo Calibration

**Calibration Parameters:**

```c
/**
 * @brief Stereo camera calibration data
 */
struct stereo_calibration {
    /* Left camera intrinsics */
    struct {
        float K[3][3];     /* Camera matrix */
        float dist[5];     /* Distortion coefficients */
    } left;
    
    /* Right camera intrinsics */
    struct {
        float K[3][3];
        float dist[5];
    } right;
    
    /* Stereo extrinsics */
    float R[3][3];         /* Rotation */
    float T[3];            /* Translation */
    
    /* Rectification */
    float R1[3][3];        /* Left rectification */
    float R2[3][3];        /* Right rectification */
    float P1[3][4];        /* Left projection */
    float P2[3][4];        /* Right projection */
    float Q[4][4];         /* Disparity-to-depth mapping */
};

/**
 * @brief Perform stereo calibration
 */
static int stereo_calibrate(
    struct stereo_calibration *calib,
    float **object_points,      /* 3D points */
    float **image_points_left,  /* 2D points in left images */
    float **image_points_right, /* 2D points in right images */
    unsigned int num_images,
    unsigned int points_per_image,
    unsigned int image_width,
    unsigned int image_height)
{
    /* This would use OpenCV's stereoCalibrate or custom implementation */
    
    /* 1. Calibrate each camera individually */
    calibrate_camera(object_points, image_points_left, num_images,
                    points_per_image, image_width, image_height,
                    calib->left.K, calib->left.dist);
    
    calibrate_camera(object_points, image_points_right, num_images,
                    points_per_image, image_width, image_height,
                    calib->right.K, calib->right.dist);
    
    /* 2. Compute stereo extrinsics (R, T) */
    stereo_calibrate_extrinsics(object_points, image_points_left,
                                image_points_right, num_images,
                                &calib->left, &calib->right,
                                calib->R, calib->T);
    
    /* 3. Compute rectification transforms */
    stereo_rectify(calib->left.K, calib->left.dist,
                  calib->right.K, calib->right.dist,
                  image_width, image_height,
                  calib->R, calib->T,
                  calib->R1, calib->R2,
                  calib->P1, calib->P2,
                  calib->Q);
    
    return 0;
}
```

#### 1.3 Stereo Rectification

**Purpose:** Transform images so epipolar lines are horizontal and aligned.

```c
/**
 * @brief Rectify stereo image pair
 */
static void stereo_rectify_images(
    uint8_t *left_in,
    uint8_t *right_in,
    uint8_t *left_out,
    uint8_t *right_out,
    unsigned int width,
    unsigned int height,
    struct stereo_calibration *calib)
{
    /* Compute rectification maps */
    float *map_left_x = malloc(width * height * sizeof(float));
    float *map_left_y = malloc(width * height * sizeof(float));
    float *map_right_x = malloc(width * height * sizeof(float));
    float *map_right_y = malloc(width * height * sizeof(float));
    
    init_undistort_rectify_map(calib->left.K, calib->left.dist,
                               calib->R1, calib->P1,
                               width, height,
                               map_left_x, map_left_y);
    
    init_undistort_rectify_map(calib->right.K, calib->right.dist,
                               calib->R2, calib->P2,
                               width, height,
                               map_right_x, map_right_y);
    
    /* Remap images */
    remap_image(left_in, left_out, width, height,
               map_left_x, map_left_y);
    
    remap_image(right_in, right_out, width, height,
                map_right_x, map_right_y);
    
    free(map_left_x);
    free(map_left_y);
    free(map_right_x);
    free(map_right_y);
}

/**
 * @brief Bilinear interpolation for remapping
 */
static void remap_image(
    uint8_t *src,
    uint8_t *dst,
    unsigned int width,
    unsigned int height,
    float *map_x,
    float *map_y)
{
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            
            float src_x = map_x[idx];
            float src_y = map_y[idx];
            
            /* Bilinear interpolation */
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            /* Boundary check */
            if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) {
                dst[idx] = 0;
                continue;
            }
            
            /* Interpolate */
            float v00 = src[y0 * width + x0];
            float v01 = src[y0 * width + x1];
            float v10 = src[y1 * width + x0];
            float v11 = src[y1 * width + x1];
            
            float v0 = v00 * (1 - fx) + v01 * fx;
            float v1 = v10 * (1 - fx) + v11 * fx;
            float v = v0 * (1 - fy) + v1 * fy;
            
            dst[idx] = (uint8_t)v;
        }
    }
}
```

### 🔹 Part 2: Stereo Matching Algorithms

#### 2.1 Block Matching (SAD/SSD)

**Sum of Absolute Differences:**

```c
/**
 * @brief Block matching stereo algorithm
 */
struct block_matching_params {
    unsigned int block_size;    /* Window size (e.g., 11x11) */
    int min_disparity;
    int max_disparity;
    int uniqueness_ratio;       /* Uniqueness check (%) */
};

/**
 * @brief Compute SAD for a block
 */
static int compute_sad(
    uint8_t *left,
    uint8_t *right,
    unsigned int width,
    unsigned int x_left,
    unsigned int y,
    unsigned int x_right,
    unsigned int block_size)
{
    int sad = 0;
    int half_block = block_size / 2;
    
    for (int dy = -half_block; dy <= half_block; dy++) {
        for (int dx = -half_block; dx <= half_block; dx++) {
            int yl = y + dy;
            int xl = x_left + dx;
            int xr = x_right + dx;
            
            if (yl < 0 || yl >= height ||
                xl < 0 || xl >= width ||
                xr < 0 || xr >= width)
                continue;
            
            int diff = abs(left[yl * width + xl] - right[yl * width + xr]);
            sad += diff;
        }
    }
    
    return sad;
}

/**
 * @brief Block matching disparity computation
 */
static void block_matching_stereo(
    uint8_t *left,
    uint8_t *right,
    int16_t *disparity,
    unsigned int width,
    unsigned int height,
    struct block_matching_params *params)
{
    int half_block = params->block_size / 2;
    
    for (unsigned int y = half_block; y < height - half_block; y++) {
        for (unsigned int x = half_block; x < width - half_block; x++) {
            int best_disparity = 0;
            int best_cost = INT_MAX;
            int second_best_cost = INT_MAX;
            
            /* Search along epipolar line (horizontal after rectification) */
            for (int d = params->min_disparity; d <= params->max_disparity; d++) {
                int x_right = x - d;
                
                if (x_right < half_block)
                    break;
                
                /* Compute matching cost */
                int cost = compute_sad(left, right, width,
                                      x, y, x_right,
                                      params->block_size);
                
                if (cost < best_cost) {
                    second_best_cost = best_cost;
                    best_cost = cost;
                    best_disparity = d;
                } else if (cost < second_best_cost) {
                    second_best_cost = cost;
                }
            }
            
            /* Uniqueness check */
            if (second_best_cost < best_cost * (100 + params->uniqueness_ratio) / 100) {
                /* Match not unique enough */
                disparity[y * width + x] = -1;
            } else {
                disparity[y * width + x] = best_disparity;
            }
        }
    }
}
```

#### 2.2 Semi-Global Matching (SGM)

**Dynamic Programming with Multiple Paths:**

```c
/**
 * @brief Semi-Global Matching
 */
struct sgm_params {
    int min_disparity;
    int max_disparity;
    int P1;  /* Small penalty for small disparity changes */
    int P2;  /* Large penalty for large disparity changes */
    unsigned int num_paths;  /* Typically 8 or 16 */
};

/**
 * @brief Compute pixel-wise matching cost
 */
static void compute_matching_costs(
    uint8_t *left,
    uint8_t *right,
    uint16_t *costs,
    unsigned int width,
    unsigned int height,
    int min_disp,
    int max_disp)
{
    int num_disp = max_disp - min_disp + 1;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            for (int d = 0; d < num_disp; d++) {
                int x_right = x - (min_disp + d);
                
                if (x_right < 0 || x_right >= width) {
                    costs[(y * width + x) * num_disp + d] = UINT16_MAX;
                    continue;
                }
                
                /* Census transform or pixel difference */
                int diff = abs(left[y * width + x] - right[y * width + x_right]);
                costs[(y * width + x) * num_disp + d] = diff;
            }
        }
    }
}

/**
 * @brief Aggregate costs along one path
 */
static void aggregate_path(
    uint16_t *costs,
    uint16_t *aggregated,
    unsigned int width,
    unsigned int height,
    int num_disp,
    int dx,
    int dy,
    struct sgm_params *params)
{
    /* Dynamic programming along path direction (dx, dy) */
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int prev_x = x - dx;
            int prev_y = y - dy;
            
            if (prev_x < 0 || prev_x >= width ||
                prev_y < 0 || prev_y >= height) {
                /* First pixel in path */
                for (int d = 0; d < num_disp; d++) {
                    unsigned int idx = (y * width + x) * num_disp + d;
                    aggregated[idx] = costs[idx];
                }
                continue;
            }
            
            /* Find minimum cost at previous pixel */
            uint16_t min_prev = UINT16_MAX;
            for (int d = 0; d < num_disp; d++) {
                unsigned int prev_idx = (prev_y * width + prev_x) * num_disp + d;
                if (aggregated[prev_idx] < min_prev)
                    min_prev = aggregated[prev_idx];
            }
            
            /* Aggregate costs with smoothness penalties */
            for (int d = 0; d < num_disp; d++) {
                unsigned int idx = (y * width + x) * num_disp + d;
                unsigned int prev_idx = (prev_y * width + prev_x) * num_disp + d;
                
                uint16_t cost = costs[idx];
                
                /* Path cost options */
                uint16_t path_costs[3];
                
                /* Same disparity */
                path_costs[0] = aggregated[prev_idx];
                
                /* Disparity +/- 1 */
                if (d > 0) {
                    path_costs[1] = aggregated[prev_idx - 1] + params->P1;
                } else {
                    path_costs[1] = UINT16_MAX;
                }
                
                if (d < num_disp - 1) {
                    path_costs[2] = aggregated[prev_idx + 1] + params->P1;
                } else {
                    path_costs[2] = UINT16_MAX;
                }
                
                /* Larger disparity changes */
                uint16_t large_change = min_prev + params->P2;
                
                /* Find minimum */
                uint16_t min_path = path_costs[0];
                if (path_costs[1] < min_path) min_path = path_costs[1];
                if (path_costs[2] < min_path) min_path = path_costs[2];
                if (large_change < min_path) min_path = large_change;
                
                aggregated[idx] = cost + min_path - min_prev;
            }
        }
    }
}

/**
 * @brief SGM disparity computation
 */
static void sgm_stereo(
    uint8_t *left,
    uint8_t *right,
    int16_t *disparity,
    unsigned int width,
    unsigned int height,
    struct sgm_params *params)
{
    int num_disp = params->max_disparity - params->min_disparity + 1;
    
    /* Compute matching costs */
    uint16_t *costs = malloc(width * height * num_disp * sizeof(uint16_t));
    compute_matching_costs(left, right, costs, width, height,
                          params->min_disparity, params->max_disparity);
    
    /* Aggregate costs from multiple paths */
    uint16_t *aggregated = calloc(width * height * num_disp, sizeof(uint16_t));
    
    /* 8 paths: horizontal, vertical, and 4 diagonals */
    int paths[][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    
    for (unsigned int p = 0; p < 8; p++) {
        uint16_t *path_costs = malloc(width * height * num_disp * sizeof(uint16_t));
        aggregate_path(costs, path_costs, width, height, num_disp,
                      paths[p][0], paths[p][1], params);
        
        /* Sum path costs */
        for (unsigned int i = 0; i < width * height * num_disp; i++) {
            aggregated[i] += path_costs[i];
        }
        
        free(path_costs);
    }
    
    /* Winner-takes-all disparity selection */
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            uint16_t min_cost = UINT16_MAX;
            int best_d = 0;
            
            for (int d = 0; d < num_disp; d++) {
                unsigned int idx = (y * width + x) * num_disp + d;
                if (aggregated[idx] < min_cost) {
                    min_cost = aggregated[idx];
                    best_d = d;
                }
            }
            
            disparity[y * width + x] = params->min_disparity + best_d;
        }
    }
    
    free(costs);
    free(aggregated);
}
```

### 🔹 Part 3: Depth Map Processing

#### 3.1 Disparity Refinement

**Sub-Pixel Interpolation:**

```c
/**
 * @brief Refine disparity to sub-pixel accuracy
 */
static float refine_disparity_subpixel(
    uint16_t *costs,
    int best_d,
    int num_disp)
{
    if (best_d <= 0 || best_d >= num_disp - 1)
        return (float)best_d;
    
    /* Parabola fitting */
    float c_prev = costs[best_d - 1];
    float c_curr = costs[best_d];
    float c_next = costs[best_d + 1];
    
    /* Sub-pixel offset */
    float offset = (c_prev - c_next) / (2.0f * (c_prev - 2.0f * c_curr + c_next));
    
    return best_d + offset;
}
```

**Median Filtering:**

```c
/**
 * @brief Median filter for disparity map
 */
static void median_filter_disparity(
    int16_t *disparity_in,
    int16_t *disparity_out,
    unsigned int width,
    unsigned int height,
    unsigned int kernel_size)
{
    int half_kernel = kernel_size / 2;
    int16_t *window = malloc(kernel_size * kernel_size * sizeof(int16_t));
    
    for (unsigned int y = half_kernel; y < height - half_kernel; y++) {
        for (unsigned int x = half_kernel; x < width - half_kernel; x++) {
            /* Collect window values */
            unsigned int count = 0;
            for (int dy = -half_kernel; dy <= half_kernel; dy++) {
                for (int dx = -half_kernel; dx <= half_kernel; dx++) {
                    int16_t val = disparity_in[(y + dy) * width + (x + dx)];
                    if (val >= 0) {  /* Valid disparity */
                        window[count++] = val;
                    }
                }
            }
            
            if (count > 0) {
                /* Sort and find median */
                qsort(window, count, sizeof(int16_t), compare_int16);
                disparity_out[y * width + x] = window[count / 2];
            } else {
                disparity_out[y * width + x] = -1;
            }
        }
    }
    
    free(window);
}
```

#### 3.2 Depth Map Generation

```c
/**
 * @brief Convert disparity to depth map
 */
static void disparity_to_depth(
    int16_t *disparity,
    float *depth,
    unsigned int width,
    unsigned int height,
    float focal_length,
    float baseline)
{
    for (unsigned int i = 0; i < width * height; i++) {
        if (disparity[i] > 0) {
            depth[i] = (focal_length * baseline) / disparity[i];
        } else {
            depth[i] = INFINITY;  /* Invalid depth */
        }
    }
}
```

#### 3.3 Point Cloud Generation

```c
/**
 * @brief Generate 3D point cloud from stereo
 */
struct point3d {
    float x, y, z;
    uint8_t r, g, b;
};

static struct point3d *generate_point_cloud(
    int16_t *disparity,
    uint8_t *color,
    unsigned int width,
    unsigned int height,
    float Q[4][4],  /* Disparity-to-depth matrix */
    unsigned int *num_points_out)
{
    /* Count valid points */
    unsigned int num_valid = 0;
    for (unsigned int i = 0; i < width * height; i++) {
        if (disparity[i] > 0)
            num_valid++;
    }
    
    struct point3d *points = malloc(num_valid * sizeof(struct point3d));
    unsigned int point_idx = 0;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = y * width + x;
            
            if (disparity[idx] <= 0)
                continue;
            
            /* Reproject to 3D using Q matrix */
            float d = disparity[idx];
            
            float X = Q[0][0] * x + Q[0][3];
            float Y = Q[1][1] * y + Q[1][3];
            float Z = Q[2][3];
            float W = Q[3][2] * d + Q[3][3];
            
            points[point_idx].x = X / W;
            points[point_idx].y = Y / W;
            points[point_idx].z = Z / W;
            
            /* Color from left image */
            points[point_idx].r = color[idx * 3 + 0];
            points[point_idx].g = color[idx * 3 + 1];
            points[point_idx].b = color[idx * 3 + 2];
            
            point_idx++;
        }
    }
    
    *num_points_out = num_valid;
    return points;
}

/**
 * @brief Save point cloud to PLY format
 */
static int save_point_cloud_ply(
    const char *filename,
    struct point3d *points,
    unsigned int num_points)
{
    FILE *f = fopen(filename, "w");
    if (!f)
        return -1;
    
    /* PLY header */
    fprintf(f, "ply\n");
    fprintf(f, "format ascii 1.0\n");
    fprintf(f, "element vertex %u\n", num_points);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "property uchar red\n");
    fprintf(f, "property uchar green\n");
    fprintf(f, "property uchar blue\n");
    fprintf(f, "end_header\n");
    
    /* Point data */
    for (unsigned int i = 0; i < num_points; i++) {
        fprintf(f, "%f %f %f %u %u %u\n",
                points[i].x, points[i].y, points[i].z,
                points[i].r, points[i].g, points[i].b);
    }
    
    fclose(f);
    return 0;
}
```

---

## 💻 Implementation Examples

### Example 1: Complete Stereo Vision Pipeline

```c
/**
 * @file stereo_pipeline.c
 * @brief Complete stereo vision processing pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct stereo_pipeline {
    /* Calibration */
    struct stereo_calibration calib;
    
    /* Parameters */
    struct sgm_params sgm_params;
    
    /* Buffers */
    uint8_t *left_rect;
    uint8_t *right_rect;
    int16_t *disparity;
    float *depth;
    
    unsigned int width;
    unsigned int height;
};

/**
 * @brief Initialize stereo pipeline
 */
static struct stereo_pipeline *stereo_pipeline_init(
    unsigned int width,
    unsigned int height,
    const char *calib_file)
{
    struct stereo_pipeline *pipeline = calloc(1, sizeof(*pipeline));
    
    pipeline->width = width;
    pipeline->height = height;
    
    /* Load calibration */
    load_stereo_calibration(calib_file, &pipeline->calib);
    
    /* Configure SGM */
    pipeline->sgm_params.min_disparity = 0;
    pipeline->sgm_params.max_disparity = 128;
    pipeline->sgm_params.P1 = 10;
    pipeline->sgm_params.P2 = 120;
    pipeline->sgm_params.num_paths = 8;
    
    /* Allocate buffers */
    pipeline->left_rect = malloc(width * height);
    pipeline->right_rect = malloc(width * height);
    pipeline->disparity = malloc(width * height * sizeof(int16_t));
    pipeline->depth = malloc(width * height * sizeof(float));
    
    return pipeline;
}

/**
 * @brief Process stereo pair
 */
static int stereo_pipeline_process(
    struct stereo_pipeline *pipeline,
    uint8_t *left_raw,
    uint8_t *right_raw)
{
    printf("Stereo Pipeline Processing:\n");
    
    /* 1. Rectification */
    printf("  [1/4] Rectifying images...\n");
    stereo_rectify_images(left_raw, right_raw,
                         pipeline->left_rect, pipeline->right_rect,
                         pipeline->width, pipeline->height,
                         &pipeline->calib);
    
    /* 2. Stereo matching */
    printf("  [2/4] Computing disparity (SGM)...\n");
    sgm_stereo(pipeline->left_rect, pipeline->right_rect,
              pipeline->disparity,
              pipeline->width, pipeline->height,
              &pipeline->sgm_params);
    
    /* 3. Disparity refinement */
    printf("  [3/4] Refining disparity...\n");
    int16_t *disparity_filtered = malloc(pipeline->width * pipeline->height * 
                                        sizeof(int16_t));
    median_filter_disparity(pipeline->disparity, disparity_filtered,
                           pipeline->width, pipeline->height, 5);
    memcpy(pipeline->disparity, disparity_filtered,
           pipeline->width * pipeline->height * sizeof(int16_t));
    free(disparity_filtered);
    
    /* 4. Depth computation */
    printf("  [4/4] Computing depth map...\n");
    float focal_length = pipeline->calib.P1[0][0];
    float baseline = -pipeline->calib.P2[0][3] / focal_length;
    
    disparity_to_depth(pipeline->disparity, pipeline->depth,
                      pipeline->width, pipeline->height,
                      focal_length, baseline);
    
    printf("Stereo processing complete\n");
    
    return 0;
}

/**
 * @brief Example usage
 */
int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <left.raw> <right.raw> <calib.yml>\n",
                argv[0]);
        return 1;
    }
    
    unsigned int width = 1280;
    unsigned int height = 720;
    
    /* Initialize pipeline */
    struct stereo_pipeline *pipeline = stereo_pipeline_init(width, height,
                                                           argv[3]);
    
    /* Load images */
    uint8_t *left = malloc(width * height);
    uint8_t *right = malloc(width * height);
    
    FILE *f_left = fopen(argv[1], "rb");
    FILE *f_right = fopen(argv[2], "rb");
    
    fread(left, 1, width * height, f_left);
    fread(right, 1, width * height, f_right);
    
    fclose(f_left);
    fclose(f_right);
    
    /* Process */
    stereo_pipeline_process(pipeline, left, right);
    
    /* Generate point cloud */
    unsigned int num_points;
    struct point3d *cloud = generate_point_cloud(
        pipeline->disparity,
        left,  /* Use left image for color */
        width, height,
        pipeline->calib.Q,
        &num_points
    );
    
    /* Save outputs */
    save_disparity_image("disparity.png", pipeline->disparity, width, height);
    save_depth_image("depth.png", pipeline->depth, width, height);
    save_point_cloud_ply("pointcloud.ply", cloud, num_points);
    
    printf("Saved outputs:\n");
    printf("  - disparity.png\n");
    printf("  - depth.png\n");
    printf("  - pointcloud.ply (%u points)\n", num_points);
    
    /* Cleanup */
    free(left);
    free(right);
    free(cloud);
    free(pipeline->left_rect);
    free(pipeline->right_rect);
    free(pipeline->disparity);
    free(pipeline->depth);
    free(pipeline);
    
    return 0;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Stereo Calibration

```python
#!/usr/bin/env python3
"""
stereo_calibration.py
Perform stereo camera calibration
"""

import numpy as np
import cv2
import glob

# Checkerboard parameters
CHECKERBOARD = (9, 6)  # Inner corners
square_size = 25.0  # mm

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left image
imgpoints_right = []  # 2D points in right image

# Load calibration images
left_images = sorted(glob.glob('calib_left/*.png'))
right_images = sorted(glob.glob('calib_right/*.png'))

print(f"Found {len(left_images)} calibration image pairs")

for left_img, right_img in zip(left_images, right_images):
    img_left = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
    
    # Find checkerboard corners
    ret_left, corners_left = cv2.findChessboardCorners(img_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(img_right, CHECKERBOARD, None)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
        
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        
        print(f"  ✓ {left_img}")
    else:
        print(f"  ✗ {left_img} - corners not found")

# Calibrate individual cameras
print("\nCalibrating left camera...")
ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_left.shape[::-1], None, None
)

print("Calibrating right camera...")
ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_right.shape[::-1], None, None
)

# Stereo calibration
print("\nPerforming stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K_left, dist_left,
    K_right, dist_right,
    img_left.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

print(f"Stereo calibration RMS error: {ret_stereo:.4f}")

# Stereo rectification
print("\nComputing rectification...")
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    K_left, dist_left,
    K_right, dist_right,
    img_left.shape[::-1],
    R, T,
    alpha=0
)

# Save calibration
calib_data = {
    'K_left': K_left,
    'dist_left': dist_left,
    'K_right': K_right,
    'dist_right': dist_right,
    'R': R,
    'T': T,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q
}

np.savez('stereo_calibration.npz', **calib_data)
print("\nCalibration saved to stereo_calibration.npz")

# Print results
baseline = -T[0] / 1000  # Convert to meters
focal_length = P1[0, 0]

print(f"\nCalibration Results:")
print(f"  Baseline: {baseline:.3f} m")
print(f"  Focal length: {focal_length:.1f} pixels")
print(f"  Min depth (disp=128): {(focal_length * baseline) / 128:.2f} m")
print(f"  Max depth (disp=1): {(focal_length * baseline) / 1:.2f} m")
```

### Lab 2: Real-Time Stereo Depth

```python
#!/usr/bin/env python3
"""
realtime_stereo_depth.py
Real-time stereo depth estimation
"""

import cv2
import numpy as np

# Load calibration
calib = np.load('stereo_calibration.npz')

# Create stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Open stereo cameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Compute rectification maps
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    calib['K_left'], calib['dist_left'], calib['R1'], calib['P1'],
    (1280, 720), cv2.CV_32FC1
)

map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    calib['K_right'], calib['dist_right'], calib['R2'], calib['P2'],
    (1280, 720), cv2.CV_32FC1
)

print("Real-time stereo depth estimation")
print("Press 'q' to quit, 's' to save point cloud")

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        break
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    # Rectify
    rect_left = cv2.remap(gray_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(gray_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    # Compute disparity
    disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0
    
    # Normalize for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Display
    cv2.imshow('Left', rect_left)
    cv2.imshow('Right', rect_right)
    cv2.imshow('Disparity', disp_color)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Generate and save point cloud
        points_3d = cv2.reprojectImageTo3D(disparity, calib['Q'])
        
        # Filter invalid points
        mask = disparity > 0
        points = points_3d[mask]
        colors = frame_left[mask]
        
        # Save PLY
        with open('pointcloud.ply', 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            for pt, col in zip(points, colors):
                f.write(f'{pt[0]} {pt[1]} {pt[2]} {col[2]} {col[1]} {col[0]}\n')
        
        print("Point cloud saved to pointcloud.ply")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
```

---

## 🐛 Debugging Techniques

### Debug 1: Calibration Quality Check

```python
#!/usr/bin/env python3
"""
check_calibration_quality.py
Verify stereo calibration quality
"""

import numpy as np
import cv2

def check_epipolar_error(calib_file, left_img, right_img):
    """Check epipolar alignment after rectification"""
    
    calib = np.load(calib_file)
    
    # Load images
    img_left = cv2.imread(left_img)
    img_right = cv2.imread(right_img)
    
    # Rectify
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        calib['K_left'], calib['dist_left'], calib['R1'], calib['P1'],
        img_left.shape[:2][::-1], cv2.CV_32FC1
    )
    
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        calib['K_right'], calib['dist_right'], calib['R2'], calib['P2'],
        img_right.shape[:2][::-1], cv2.CV_32FC1
    )
    
    rect_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    
    # Draw epipolar lines
    height = rect_left.shape[0]
    for y in range(0, height, 30):
        cv2.line(rect_left, (0, y), (rect_left.shape[1], y), (0, 255, 0), 1)
        cv2.line(rect_right, (0, y), (rect_right.shape[1], y), (0, 255, 0), 1)
    
    # Display
    combined = np.hstack([rect_left, rect_right])
    cv2.imshow('Epipolar Lines', combined)
    cv2.waitKey(0)
    
    print("Check if corresponding points lie on same horizontal lines")

if __name__ == '__main__':
    check_calibration_quality('stereo_calibration.npz',
                              'test_left.png', 'test_right.png')
```

### Debug 2: Disparity Map Quality

```c
/**
 * @brief Analyze disparity map quality
 */
struct disparity_quality {
    float fill_rate;        /* Percentage of valid disparities */
    float mean_disparity;
    float std_disparity;
    unsigned int num_outliers;
};

static struct disparity_quality analyze_disparity_quality(
    int16_t *disparity,
    unsigned int width,
    unsigned int height)
{
    struct disparity_quality quality = {0};
    
    unsigned int valid_count = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (unsigned int i = 0; i < width * height; i++) {
        if (disparity[i] > 0) {
            valid_count++;
            sum += disparity[i];
            sum_sq += disparity[i] * disparity[i];
        }
    }
    
    quality.fill_rate = (float)valid_count / (width * height) * 100.0f;
    
    if (valid_count > 0) {
        quality.mean_disparity = sum / valid_count;
        float variance = (sum_sq / valid_count) - 
                        (quality.mean_disparity * quality.mean_disparity);
        quality.std_disparity = sqrtf(variance);
    }
    
    /* Count outliers (disparity changes > threshold) */
    for (unsigned int y = 1; y < height - 1; y++) {
        for (unsigned int x = 1; x < width - 1; x++) {
            int16_t center = disparity[y * width + x];
            if (center <= 0)
                continue;
            
            int16_t neighbors[4] = {
                disparity[(y-1) * width + x],
                disparity[(y+1) * width + x],
                disparity[y * width + (x-1)],
                disparity[y * width + (x+1)]
            };
            
            for (int i = 0; i < 4; i++) {
                if (neighbors[i] > 0 && abs(center - neighbors[i]) > 20) {
                    quality.num_outliers++;
                    break;
                }
            }
        }
    }
    
    return quality;
}
```

---

## ⚡ Performance Optimization

### Optimization 1: GPU-Accelerated SGM

```cuda
/**
 * @file sgm_cuda.cu
 * @brief CUDA-accelerated Semi-Global Matching
 */

__global__ void compute_costs_kernel(
    const uint8_t *left,
    const uint8_t *right,
    uint16_t *costs,
    unsigned int width,
    unsigned int height,
    int min_disp,
    int num_disp)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    uint8_t left_val = left[y * width + x];
    
    for (int d = 0; d < num_disp; d++) {
        int x_right = x - (min_disp + d);
        
        if (x_right < 0 || x_right >= width) {
            costs[(y * width + x) * num_disp + d] = UINT16_MAX;
        } else {
            uint8_t right_val = right[y * width + x_right];
            costs[(y * width + x) * num_disp + d] = abs(left_val - right_val);
        }
    }
}

extern "C" void sgm_cuda(
    uint8_t *h_left,
    uint8_t *h_right,
    int16_t *h_disparity,
    unsigned int width,
    unsigned int height,
    int min_disp,
    int max_disp)
{
    int num_disp = max_disp - min_disp + 1;
    
    /* Allocate device memory */
    uint8_t *d_left, *d_right;
    uint16_t *d_costs;
    int16_t *d_disparity;
    
    size_t img_size = width * height;
    size_t cost_size = img_size * num_disp * sizeof(uint16_t);
    
    cudaMalloc(&d_left, img_size);
    cudaMalloc(&d_right, img_size);
    cudaMalloc(&d_costs, cost_size);
    cudaMalloc(&d_disparity, img_size * sizeof(int16_t));
    
    /* Copy input */
    cudaMemcpy(d_left, h_left, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, h_right, img_size, cudaMemcpyHostToDevice);
    
    /* Compute costs */
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    compute_costs_kernel<<<grid, block>>>(
        d_left, d_right, d_costs,
        width, height, min_disp, num_disp
    );
    
    /* Path aggregation (simplified - full implementation would be more complex) */
    /* ... */
    
    /* Copy result */
    cudaMemcpy(h_disparity, d_disparity, img_size * sizeof(int16_t),
              cudaMemcpyDeviceToHost);
    
    /* Cleanup */
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_costs);
    cudaFree(d_disparity);
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain epipolar geometry and why rectification simplifies stereo matching.**

2. **What is the relationship between disparity and depth? Derive the formula.**

3. **Compare block matching vs SGM: advantages and disadvantages.**

4. **Why is sub-pixel disparity refinement important?**

5. **Calculate minimum detectable depth for:**
   - Baseline: 120mm
   - Focal length: 800 pixels
   - Min disparity: 1 pixel

### Practical Challenges

1. **Implement Census transform for robust stereo matching.**

2. **Debug a stereo system with poor depth accuracy in one region.**

3. **Optimize SGM to run at 30fps for 1280x720.**

4. **Design a stereo rig for 0.5m to 10m depth range.**

---

## 📚 Further Reading & Resources

### Books
- "Multiple View Geometry in Computer Vision" - Hartley & Zisserman
- "Computer Vision: Algorithms and Applications" - Szeliski

### Papers
- "Semi-Global Matching" - Hirschmüller
- "Efficient Large-Scale Stereo Matching" - Geiger et al.

### Tools
- **OpenCV:** cv::StereoBM, cv::StereoSGBM
- **PCL:** Point Cloud Library
- **MeshLab:** Point cloud visualization

---

## 🎓 Summary

Today we covered:
- ✅ Stereo vision fundamentals and epipolar geometry
- ✅ Stereo calibration and rectification
- ✅ Block matching and Semi-Global Matching algorithms
- ✅ Disparity refinement and depth map generation
- ✅ Point cloud creation and visualization
- ✅ Complete stereo vision pipeline implementation
- ✅ GPU acceleration for real-time performance

**Key Takeaways:**
1. Stereo vision provides 3D depth from 2D images
2. Calibration and rectification are critical for accuracy
3. SGM provides better quality than block matching
4. Real-time performance requires GPU acceleration

**Next:** Day 11 - Camera Calibration and Geometric Correction

---

**Day 10 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
