# Day 11: Camera Calibration and Geometric Correction
## Phase 3: Camera Systems & ISP | Week 2: Advanced Camera Features

---

## 🎯 Learning Objectives
1. **Understand** camera calibration theory and pinhole camera model
2. **Implement** intrinsic and extrinsic calibration algorithms
3. **Configure** lens distortion correction (radial and tangential)
4. **Develop** automatic calibration and validation tools
5. **Debug** calibration errors and geometric artifacts
6. **Optimize** real-time undistortion performance

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera with lens, calibration pattern (checkerboard)
*   **Software:** OpenCV, calibration tools, image processing libraries
*   **Knowledge:** Linear algebra, projective geometry, optimization
*   **Tools:** Calibration patterns, measurement tools

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Camera Model Fundamentals

#### 1.1 Pinhole Camera Model

**Projection Equation:**

```
World coordinates (X, Y, Z) → Image coordinates (u, v)

[u]   [fx  0  cx] [r11 r12 r13 tx] [X]
[v] = [ 0 fy  cy] [r21 r22 r23 ty] [Y]
[1]   [ 0  0   1] [r31 r32 r33 tz] [Z]
                                    [1]

Where:
- (fx, fy): focal length in pixels
- (cx, cy): principal point (image center)
- R = [rij]: rotation matrix
- t = [tx, ty, tz]: translation vector
```

**Intrinsic Parameters:**

```c
/**
 * @brief Camera intrinsic parameters
 */
struct camera_intrinsics {
    float fx, fy;      /* Focal length in pixels */
    float cx, cy;      /* Principal point */
    float skew;        /* Skew coefficient (usually 0) */
    
    /* Camera matrix K */
    float K[3][3];     /* [fx  s  cx]
                          [ 0 fy  cy]
                          [ 0  0   1] */
};

/**
 * @brief Initialize camera matrix
 */
static void init_camera_matrix(
    struct camera_intrinsics *intrinsics)
{
    intrinsics->K[0][0] = intrinsics->fx;
    intrinsics->K[0][1] = intrinsics->skew;
    intrinsics->K[0][2] = intrinsics->cx;
    
    intrinsics->K[1][0] = 0.0f;
    intrinsics->K[1][1] = intrinsics->fy;
    intrinsics->K[1][2] = intrinsics->cy;
    
    intrinsics->K[2][0] = 0.0f;
    intrinsics->K[2][1] = 0.0f;
    intrinsics->K[2][2] = 1.0f;
}
```

**Extrinsic Parameters:**

```c
/**
 * @brief Camera extrinsic parameters
 */
struct camera_extrinsics {
    float R[3][3];     /* Rotation matrix */
    float t[3];        /* Translation vector */
    
    /* Combined [R|t] matrix */
    float RT[3][4];
};

/**
 * @brief Rodrigues rotation vector to matrix
 */
static void rodrigues_to_matrix(
    const float rvec[3],
    float R[3][3])
{
    float theta = sqrtf(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]);
    
    if (theta < 1e-6f) {
        /* Identity */
        memset(R, 0, sizeof(float) * 9);
        R[0][0] = R[1][1] = R[2][2] = 1.0f;
        return;
    }
    
    /* Normalized axis */
    float k[3] = {rvec[0]/theta, rvec[1]/theta, rvec[2]/theta};
    
    float c = cosf(theta);
    float s = sinf(theta);
    float c1 = 1.0f - c;
    
    /* Rodrigues formula */
    R[0][0] = c + k[0]*k[0]*c1;
    R[0][1] = k[0]*k[1]*c1 - k[2]*s;
    R[0][2] = k[0]*k[2]*c1 + k[1]*s;
    
    R[1][0] = k[1]*k[0]*c1 + k[2]*s;
    R[1][1] = c + k[1]*k[1]*c1;
    R[1][2] = k[1]*k[2]*c1 - k[0]*s;
    
    R[2][0] = k[2]*k[0]*c1 - k[1]*s;
    R[2][1] = k[2]*k[1]*c1 + k[0]*s;
    R[2][2] = c + k[2]*k[2]*c1;
}
```

#### 1.2 Lens Distortion Model

**Radial Distortion:**

```
x_distorted = x_undistorted × (1 + k1×r² + k2×r⁴ + k3×r⁶)
y_distorted = y_undistorted × (1 + k1×r² + k2×r⁴ + k3×r⁶)

Where:
- r² = x² + y²
- k1, k2, k3: radial distortion coefficients
```

**Tangential Distortion:**

```
x_distorted = x_undistorted + [2×p1×x×y + p2×(r² + 2×x²)]
y_distorted = y_undistorted + [p1×(r² + 2×y²) + 2×p2×x×y]

Where:
- p1, p2: tangential distortion coefficients
```

**Combined Distortion Model:**

```c
/**
 * @brief Lens distortion parameters
 */
struct lens_distortion {
    float k1, k2, k3;  /* Radial distortion */
    float p1, p2;      /* Tangential distortion */
    float k4, k5, k6;  /* Higher-order radial (optional) */
};

/**
 * @brief Apply lens distortion
 */
static void apply_distortion(
    float x_norm,
    float y_norm,
    struct lens_distortion *dist,
    float *x_dist,
    float *y_dist)
{
    float r2 = x_norm*x_norm + y_norm*y_norm;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    
    /* Radial distortion */
    float radial = 1.0f + dist->k1*r2 + dist->k2*r4 + dist->k3*r6;
    
    /* Tangential distortion */
    float dx_tangential = 2.0f*dist->p1*x_norm*y_norm + 
                         dist->p2*(r2 + 2.0f*x_norm*x_norm);
    float dy_tangential = dist->p1*(r2 + 2.0f*y_norm*y_norm) + 
                         2.0f*dist->p2*x_norm*y_norm;
    
    /* Combined */
    *x_dist = x_norm * radial + dx_tangential;
    *y_dist = y_norm * radial + dy_tangential;
}

/**
 * @brief Remove lens distortion (iterative)
 */
static void remove_distortion(
    float x_dist,
    float y_dist,
    struct lens_distortion *dist,
    float *x_undist,
    float *y_undist)
{
    /* Iterative Newton-Raphson */
    float x = x_dist;
    float y = y_dist;
    
    for (int iter = 0; iter < 5; iter++) {
        float x_d, y_d;
        apply_distortion(x, y, dist, &x_d, &y_d);
        
        float dx = x_dist - x_d;
        float dy = y_dist - y_d;
        
        x += dx;
        y += dy;
    }
    
    *x_undist = x;
    *y_undist = y;
}
```

### 🔹 Part 2: Calibration Algorithms

#### 2.1 Zhang's Method

**Calibration Pattern:**

```c
/**
 * @brief Calibration pattern (checkerboard)
 */
struct calibration_pattern {
    unsigned int width;   /* Number of inner corners horizontally */
    unsigned int height;  /* Number of inner corners vertically */
    float square_size;    /* Size of each square (mm) */
};

/**
 * @brief Generate 3D object points for pattern
 */
static void generate_object_points(
    struct calibration_pattern *pattern,
    float *object_points)
{
    unsigned int idx = 0;
    
    for (unsigned int y = 0; y < pattern->height; y++) {
        for (unsigned int x = 0; x < pattern->width; x++) {
            object_points[idx++] = x * pattern->square_size;
            object_points[idx++] = y * pattern->square_size;
            object_points[idx++] = 0.0f;  /* Z = 0 (planar pattern) */
        }
    }
}
```

**Homography Estimation:**

```c
/**
 * @brief Estimate homography from point correspondences
 */
static int estimate_homography(
    const float *object_points,  /* 2D points in pattern */
    const float *image_points,   /* 2D points in image */
    unsigned int num_points,
    float H[3][3])
{
    /* Build matrix A for homogeneous system Ah = 0 */
    float *A = malloc(2 * num_points * 9 * sizeof(float));
    
    for (unsigned int i = 0; i < num_points; i++) {
        float X = object_points[i*2 + 0];
        float Y = object_points[i*2 + 1];
        float u = image_points[i*2 + 0];
        float v = image_points[i*2 + 1];
        
        /* Row 2i */
        A[2*i*9 + 0] = -X;
        A[2*i*9 + 1] = -Y;
        A[2*i*9 + 2] = -1.0f;
        A[2*i*9 + 3] = 0.0f;
        A[2*i*9 + 4] = 0.0f;
        A[2*i*9 + 5] = 0.0f;
        A[2*i*9 + 6] = u*X;
        A[2*i*9 + 7] = u*Y;
        A[2*i*9 + 8] = u;
        
        /* Row 2i+1 */
        A[(2*i+1)*9 + 0] = 0.0f;
        A[(2*i+1)*9 + 1] = 0.0f;
        A[(2*i+1)*9 + 2] = 0.0f;
        A[(2*i+1)*9 + 3] = -X;
        A[(2*i+1)*9 + 4] = -Y;
        A[(2*i+1)*9 + 5] = -1.0f;
        A[(2*i+1)*9 + 6] = v*X;
        A[(2*i+1)*9 + 7] = v*Y;
        A[(2*i+1)*9 + 8] = v;
    }
    
    /* Solve using SVD: A = U×S×V^T, solution is last column of V */
    /* This would use LAPACK or similar library */
    svd_solve(A, 2*num_points, 9, H);
    
    free(A);
    
    return 0;
}
```

**Intrinsic Parameter Estimation:**

```c
/**
 * @brief Estimate intrinsic parameters from homographies
 */
static int estimate_intrinsics(
    float **homographies,
    unsigned int num_views,
    struct camera_intrinsics *intrinsics)
{
    /* Build constraint matrix V from homographies */
    float *V = malloc(2 * num_views * 6 * sizeof(float));
    
    for (unsigned int i = 0; i < num_views; i++) {
        float *H = homographies[i];
        
        /* v12^T */
        float v12[6] = {
            H[0]*H[1],
            H[0]*H[4] + H[3]*H[1],
            H[3]*H[4],
            H[6]*H[1] + H[0]*H[7],
            H[6]*H[4] + H[3]*H[7],
            H[6]*H[7]
        };
        
        /* v11^T - v22^T */
        float v11_v22[6] = {
            H[0]*H[0] - H[1]*H[1],
            2.0f*(H[0]*H[3] - H[1]*H[4]),
            H[3]*H[3] - H[4]*H[4],
            2.0f*(H[0]*H[6] - H[1]*H[7]),
            2.0f*(H[3]*H[6] - H[4]*H[7]),
            H[6]*H[6] - H[7]*H[7]
        };
        
        memcpy(&V[2*i*6], v12, 6*sizeof(float));
        memcpy(&V[(2*i+1)*6], v11_v22, 6*sizeof(float));
    }
    
    /* Solve for b = [B11, B12, B22, B13, B23, B33]^T */
    float b[6];
    svd_solve(V, 2*num_views, 6, b);
    
    /* Extract intrinsic parameters from b */
    float B11 = b[0], B12 = b[1], B22 = b[2];
    float B13 = b[3], B23 = b[4], B33 = b[5];
    
    float v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12);
    float lambda = B33 - (B13*B13 + v0*(B12*B13 - B11*B23)) / B11;
    float alpha = sqrtf(lambda / B11);
    float beta = sqrtf(lambda * B11 / (B11*B22 - B12*B12));
    float gamma = -B12 * alpha*alpha * beta / lambda;
    float u0 = gamma*v0/beta - B13*alpha*alpha/lambda;
    
    intrinsics->fx = alpha;
    intrinsics->fy = beta;
    intrinsics->cx = u0;
    intrinsics->cy = v0;
    intrinsics->skew = gamma;
    
    init_camera_matrix(intrinsics);
    
    free(V);
    
    return 0;
}
```

#### 2.2 Nonlinear Refinement

**Levenberg-Marquardt Optimization:**

```c
/**
 * @brief Refine calibration using bundle adjustment
 */
struct calibration_params {
    struct camera_intrinsics intrinsics;
    struct lens_distortion distortion;
    float **extrinsics;  /* Rotation and translation for each view */
    unsigned int num_views;
};

/**
 * @brief Compute reprojection error
 */
static float compute_reprojection_error(
    const float *object_points,
    const float *image_points,
    unsigned int num_points,
    struct calibration_params *params,
    unsigned int view_idx)
{
    float total_error = 0.0f;
    
    for (unsigned int i = 0; i < num_points; i++) {
        /* 3D point */
        float X = object_points[i*3 + 0];
        float Y = object_points[i*3 + 1];
        float Z = object_points[i*3 + 2];
        
        /* Transform to camera coordinates */
        float *R = params->extrinsics[view_idx];
        float *t = &params->extrinsics[view_idx][9];
        
        float Xc = R[0]*X + R[1]*Y + R[2]*Z + t[0];
        float Yc = R[3]*X + R[4]*Y + R[5]*Z + t[1];
        float Zc = R[6]*X + R[7]*Y + R[8]*Z + t[2];
        
        /* Project to normalized image plane */
        float x = Xc / Zc;
        float y = Yc / Zc;
        
        /* Apply distortion */
        float x_dist, y_dist;
        apply_distortion(x, y, &params->distortion, &x_dist, &y_dist);
        
        /* Project to pixel coordinates */
        float u = params->intrinsics.fx * x_dist + params->intrinsics.cx;
        float v = params->intrinsics.fy * y_dist + params->intrinsics.cy;
        
        /* Compute error */
        float du = u - image_points[i*2 + 0];
        float dv = v - image_points[i*2 + 1];
        
        total_error += sqrtf(du*du + dv*dv);
    }
    
    return total_error / num_points;
}

/**
 * @brief Refine parameters using Levenberg-Marquardt
 */
static int refine_calibration(
    float **object_points,
    float **image_points,
    unsigned int *num_points_per_view,
    struct calibration_params *params)
{
    /* This would use a nonlinear optimization library like Ceres or levmar */
    /* Simplified example */
    
    float lambda = 0.001f;  /* LM damping parameter */
    
    for (int iter = 0; iter < 100; iter++) {
        /* Compute Jacobian and residuals */
        /* Update parameters */
        /* Check convergence */
        
        float total_error = 0.0f;
        for (unsigned int v = 0; v < params->num_views; v++) {
            total_error += compute_reprojection_error(
                object_points[v],
                image_points[v],
                num_points_per_view[v],
                params,
                v
            );
        }
        
        printf("Iteration %d: RMS error = %.4f pixels\n",
               iter, total_error / params->num_views);
        
        if (total_error / params->num_views < 0.5f)
            break;
    }
    
    return 0;
}
```

### 🔹 Part 3: Undistortion and Rectification

#### 3.1 Undistortion Map Generation

```c
/**
 * @brief Generate undistortion lookup tables
 */
static void init_undistort_map(
    struct camera_intrinsics *intrinsics,
    struct lens_distortion *distortion,
    unsigned int width,
    unsigned int height,
    float *map_x,
    float *map_y)
{
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            unsigned int idx = v * width + u;
            
            /* Normalize pixel coordinates */
            float x_norm = (u - intrinsics->cx) / intrinsics->fx;
            float y_norm = (v - intrinsics->cy) / intrinsics->fy;
            
            /* Remove distortion */
            float x_undist, y_undist;
            remove_distortion(x_norm, y_norm, distortion,
                            &x_undist, &y_undist);
            
            /* Convert back to pixel coordinates */
            map_x[idx] = intrinsics->fx * x_undist + intrinsics->cx;
            map_y[idx] = intrinsics->fy * y_undist + intrinsics->cy;
        }
    }
}

/**
 * @brief Apply undistortion using lookup tables
 */
static void undistort_image(
    uint8_t *src,
    uint8_t *dst,
    unsigned int width,
    unsigned int height,
    float *map_x,
    float *map_y)
{
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            unsigned int idx = v * width + u;
            
            float src_x = map_x[idx];
            float src_y = map_y[idx];
            
            /* Bilinear interpolation */
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) {
                dst[idx] = 0;
                continue;
            }
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
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

---

## 💻 Implementation Examples

### Example 1: Complete Calibration Tool

```c
/**
 * @file camera_calibration.c
 * @brief Complete camera calibration implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glob.h>

struct calibration_data {
    /* Pattern */
    struct calibration_pattern pattern;
    
    /* Captured data */
    unsigned int num_images;
    float **object_points;
    float **image_points;
    unsigned int *num_points_per_image;
    
    /* Results */
    struct camera_intrinsics intrinsics;
    struct lens_distortion distortion;
    float **extrinsics;
    
    /* Quality metrics */
    float rms_error;
};

/**
 * @brief Detect calibration pattern in image
 */
static int detect_pattern(
    uint8_t *image,
    unsigned int width,
    unsigned int height,
    struct calibration_pattern *pattern,
    float *corners_out)
{
    /* This would use corner detection algorithm */
    /* Simplified: assume corners are detected */
    
    unsigned int num_corners = pattern->width * pattern->height;
    
    /* Find checkerboard corners using Harris or similar */
    /* Refine to sub-pixel accuracy */
    
    return num_corners;
}

/**
 * @brief Perform camera calibration
 */
static int calibrate_camera(
    struct calibration_data *calib)
{
    printf("Camera Calibration\n");
    printf("==================\n\n");
    
    /* 1. Initial intrinsic estimation */
    printf("[1/3] Estimating intrinsic parameters...\n");
    
    float **homographies = malloc(calib->num_images * sizeof(float*));
    for (unsigned int i = 0; i < calib->num_images; i++) {
        homographies[i] = malloc(9 * sizeof(float));
        estimate_homography(calib->object_points[i],
                          calib->image_points[i],
                          calib->num_points_per_image[i],
                          (float(*)[3])homographies[i]);
    }
    
    estimate_intrinsics(homographies, calib->num_images,
                       &calib->intrinsics);
    
    printf("  fx = %.2f, fy = %.2f\n",
           calib->intrinsics.fx, calib->intrinsics.fy);
    printf("  cx = %.2f, cy = %.2f\n",
           calib->intrinsics.cx, calib->intrinsics.cy);
    
    /* 2. Initial extrinsic estimation */
    printf("\n[2/3] Estimating extrinsic parameters...\n");
    
    calib->extrinsics = malloc(calib->num_images * sizeof(float*));
    for (unsigned int i = 0; i < calib->num_images; i++) {
        calib->extrinsics[i] = malloc(12 * sizeof(float));
        /* Estimate R and t from homography */
    }
    
    /* 3. Nonlinear refinement */
    printf("\n[3/3] Refining all parameters...\n");
    
    struct calibration_params params = {
        .intrinsics = calib->intrinsics,
        .distortion = calib->distortion,
        .extrinsics = calib->extrinsics,
        .num_views = calib->num_images
    };
    
    refine_calibration(calib->object_points,
                      calib->image_points,
                      calib->num_points_per_image,
                      &params);
    
    calib->intrinsics = params.intrinsics;
    calib->distortion = params.distortion;
    
    /* 4. Compute final RMS error */
    float total_error = 0.0f;
    for (unsigned int i = 0; i < calib->num_images; i++) {
        total_error += compute_reprojection_error(
            calib->object_points[i],
            calib->image_points[i],
            calib->num_points_per_image[i],
            &params,
            i
        );
    }
    calib->rms_error = total_error / calib->num_images;
    
    printf("\nCalibration complete!\n");
    printf("  RMS reprojection error: %.4f pixels\n", calib->rms_error);
    
    /* Cleanup */
    for (unsigned int i = 0; i < calib->num_images; i++) {
        free(homographies[i]);
    }
    free(homographies);
    
    return 0;
}

/**
 * @brief Save calibration to file
 */
static int save_calibration(
    const char *filename,
    struct calibration_data *calib)
{
    FILE *f = fopen(filename, "w");
    if (!f)
        return -1;
    
    fprintf(f, "# Camera Calibration Data\n");
    fprintf(f, "# RMS Error: %.4f pixels\n\n", calib->rms_error);
    
    fprintf(f, "camera_matrix:\n");
    fprintf(f, "  fx: %.6f\n", calib->intrinsics.fx);
    fprintf(f, "  fy: %.6f\n", calib->intrinsics.fy);
    fprintf(f, "  cx: %.6f\n", calib->intrinsics.cx);
    fprintf(f, "  cy: %.6f\n", calib->intrinsics.cy);
    
    fprintf(f, "\ndistortion_coefficients:\n");
    fprintf(f, "  k1: %.6f\n", calib->distortion.k1);
    fprintf(f, "  k2: %.6f\n", calib->distortion.k2);
    fprintf(f, "  p1: %.6f\n", calib->distortion.p1);
    fprintf(f, "  p2: %.6f\n", calib->distortion.p2);
    fprintf(f, "  k3: %.6f\n", calib->distortion.k3);
    
    fclose(f);
    
    printf("\nCalibration saved to %s\n", filename);
    
    return 0;
}

/**
 * @brief Example usage
 */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_pattern>\n", argv[0]);
        fprintf(stderr, "Example: %s \"calib_*.png\"\n", argv[0]);
        return 1;
    }
    
    /* Initialize calibration data */
    struct calibration_data calib = {0};
    calib.pattern.width = 9;
    calib.pattern.height = 6;
    calib.pattern.square_size = 25.0f;  /* 25mm squares */
    
    /* Load calibration images */
    glob_t glob_result;
    glob(argv[1], GLOB_TILDE, NULL, &glob_result);
    
    calib.num_images = glob_result.gl_pathc;
    printf("Found %u calibration images\n\n", calib.num_images);
    
    calib.object_points = malloc(calib.num_images * sizeof(float*));
    calib.image_points = malloc(calib.num_images * sizeof(float*));
    calib.num_points_per_image = malloc(calib.num_images * sizeof(unsigned int));
    
    unsigned int num_corners = calib.pattern.width * calib.pattern.height;
    
    for (unsigned int i = 0; i < calib.num_images; i++) {
        /* Load image */
        uint8_t *image = load_image(glob_result.gl_pathv[i],
                                   &width, &height);
        
        /* Generate object points */
        calib.object_points[i] = malloc(num_corners * 3 * sizeof(float));
        generate_object_points(&calib.pattern, calib.object_points[i]);
        
        /* Detect pattern */
        calib.image_points[i] = malloc(num_corners * 2 * sizeof(float));
        int detected = detect_pattern(image, width, height,
                                     &calib.pattern,
                                     calib.image_points[i]);
        
        if (detected == num_corners) {
            calib.num_points_per_image[i] = num_corners;
            printf("  ✓ %s\n", glob_result.gl_pathv[i]);
        } else {
            printf("  ✗ %s - pattern not found\n", glob_result.gl_pathv[i]);
            calib.num_points_per_image[i] = 0;
        }
        
        free(image);
    }
    
    globfree(&glob_result);
    
    /* Perform calibration */
    calibrate_camera(&calib);
    
    /* Save results */
    save_calibration("camera_calibration.yml", &calib);
    
    /* Generate undistortion maps */
    float *map_x = malloc(width * height * sizeof(float));
    float *map_y = malloc(width * height * sizeof(float));
    
    init_undistort_map(&calib.intrinsics, &calib.distortion,
                      width, height, map_x, map_y);
    
    /* Save maps for runtime use */
    save_undistort_maps("undistort_maps.bin", map_x, map_y,
                       width, height);
    
    /* Cleanup */
    for (unsigned int i = 0; i < calib.num_images; i++) {
        free(calib.object_points[i]);
        free(calib.image_points[i]);
        free(calib.extrinsics[i]);
    }
    free(calib.object_points);
    free(calib.image_points);
    free(calib.num_points_per_image);
    free(calib.extrinsics);
    free(map_x);
    free(map_y);
    
    return 0;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Camera Calibration

```python
#!/usr/bin/env python3
"""
camera_calibration.py
Perform camera calibration using OpenCV
"""

import numpy as np
import cv2
import glob

# Checkerboard dimensions
CHECKERBOARD = (9, 6)
square_size = 25.0  # mm

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images
images = glob.glob('calib_*.png')
print(f"Found {len(images)} calibration images\n")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(100)
        
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} - corners not found")

cv2.destroyAllWindows()

# Calibrate camera
print("\nPerforming calibration...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"\nCalibration Results:")
print(f"  RMS reprojection error: {ret:.4f} pixels")
print(f"\nCamera Matrix:")
print(mtx)
print(f"\nDistortion Coefficients:")
print(dist.ravel())

# Save calibration
np.savez('camera_calibration.npz',
         mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, rms=ret)

print("\nCalibration saved to camera_calibration.npz")

# Test undistortion
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]

# Get optimal new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Display comparison
comparison = np.hstack([test_img, cv2.resize(dst, (test_img.shape[1], test_img.shape[0]))])
cv2.imwrite('undistortion_comparison.png', comparison)
print("Undistortion comparison saved to undistortion_comparison.png")
```

### Lab 2: Calibration Quality Assessment

```python
#!/usr/bin/env python3
"""
assess_calibration_quality.py
Evaluate calibration quality
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def assess_calibration(calib_file):
    """Assess calibration quality"""
    
    # Load calibration
    calib = np.load(calib_file)
    mtx = calib['mtx']
    dist = calib['dist']
    rvecs = calib['rvecs']
    tvecs = calib['tvecs']
    
    print("Calibration Quality Assessment")
    print("=" * 50)
    
    # 1. Reprojection errors
    print("\n1. Reprojection Errors:")
    errors = []
    
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Project points
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
        
        # Calculate error
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        
        print(f"  Image {i}: {error:.4f} pixels")
    
    print(f"\n  Mean error: {np.mean(errors):.4f} pixels")
    print(f"  Std dev: {np.std(errors):.4f} pixels")
    print(f"  Max error: {np.max(errors):.4f} pixels")
    
    # 2. Distortion magnitude
    print("\n2. Distortion Coefficients:")
    k1, k2, p1, p2, k3 = dist.ravel()
    print(f"  k1 (radial): {k1:.6f}")
    print(f"  k2 (radial): {k2:.6f}")
    print(f"  k3 (radial): {k3:.6f}")
    print(f"  p1 (tangential): {p1:.6f}")
    print(f"  p2 (tangential): {p2:.6f}")
    
    # 3. Field of view
    print("\n3. Field of View:")
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    
    # Assuming image size from calibration
    width, height = 1920, 1080
    
    fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    
    print(f"  Horizontal FOV: {fov_x:.2f}°")
    print(f"  Vertical FOV: {fov_y:.2f}°")
    
    # 4. Principal point offset
    print("\n4. Principal Point:")
    print(f"  Center: ({width/2}, {height/2})")
    print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
    print(f"  Offset: ({cx - width/2:.2f}, {cy - height/2:.2f}) pixels")
    
    # 5. Aspect ratio
    aspect_ratio = fy / fx
    print(f"\n5. Pixel Aspect Ratio: {aspect_ratio:.6f}")
    if abs(aspect_ratio - 1.0) > 0.01:
        print("  WARNING: Non-square pixels detected!")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(errors)), errors)
    plt.xlabel('Image Index')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Calibration Reprojection Errors')
    plt.axhline(y=np.mean(errors), color='r', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_errors.png')
    print("\nError plot saved to calibration_errors.png")

if __name__ == '__main__':
    assess_calibration('camera_calibration.npz')
```

---

## 🐛 Debugging Techniques

### Debug 1: Poor Calibration Quality

**Symptoms:** High RMS error, distorted undistorted images.

**Checklist:**

```python
#!/usr/bin/env python3
"""
debug_calibration.py
Debug calibration issues
"""

def debug_calibration_quality(images, objpoints, imgpoints):
    """Debug calibration quality issues"""
    
    print("Calibration Debug Checklist:")
    print("=" * 50)
    
    # 1. Check number of images
    print(f"\n1. Number of images: {len(images)}")
    if len(images) < 10:
        print("  ⚠ WARNING: Less than 10 images. Recommended: 15-20")
    
    # 2. Check pattern coverage
    print("\n2. Pattern Coverage:")
    all_corners = np.vstack(imgpoints)
    min_x, min_y = all_corners.min(axis=0).ravel()
    max_x, max_y = all_corners.max(axis=0).ravel()
    
    print(f"  X range: {min_x:.0f} - {max_x:.0f}")
    print(f"  Y range: {min_y:.0f} - {max_y:.0f}")
    
    # Should cover most of image
    if min_x > 100 or min_y > 100:
        print("  ⚠ WARNING: Pattern doesn't reach image edges")
    
    # 3. Check pattern orientations
    print("\n3. Pattern Orientations:")
    # Analyze rotation vectors
    # Should have variety of orientations
    
    # 4. Check for outliers
    print("\n4. Outlier Detection:")
    # Detect images with unusually high error
    
    # 5. Check pattern detection quality
    print("\n5. Pattern Detection Quality:")
    for i, corners in enumerate(imgpoints):
        # Check corner spacing consistency
        dists = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        std_dev = np.std(dists)
        if std_dev > 5.0:
            print(f"  ⚠ Image {i}: Inconsistent corner spacing (std={std_dev:.2f})")

if __name__ == '__main__':
    # Load calibration data
    debug_calibration_quality(images, objpoints, imgpoints)
```

### Debug 2: Geometric Distortion Artifacts

```c
/**
 * @brief Visualize distortion field
 */
static void visualize_distortion(
    struct lens_distortion *dist,
    unsigned int width,
    unsigned int height,
    const char *output_file)
{
    uint8_t *vis = malloc(width * height * 3);
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            /* Normalized coordinates */
            float x_norm = (x - width/2.0f) / (width/2.0f);
            float y_norm = (y - height/2.0f) / (height/2.0f);
            
            /* Apply distortion */
            float x_dist, y_dist;
            apply_distortion(x_norm, y_norm, dist, &x_dist, &y_dist);
            
            /* Compute displacement */
            float dx = (x_dist - x_norm) * width/2.0f;
            float dy = (y_dist - y_norm) * height/2.0f;
            
            float magnitude = sqrtf(dx*dx + dy*dy);
            
            /* Color code by magnitude */
            unsigned int idx = (y * width + x) * 3;
            uint8_t color = (uint8_t)(magnitude * 10);  /* Scale for visibility */
            
            vis[idx + 0] = color;  /* R */
            vis[idx + 1] = 0;      /* G */
            vis[idx + 2] = 255 - color;  /* B */
        }
    }
    
    save_image(output_file, vis, width, height);
    free(vis);
}
```

---

## ⚡ Performance Optimization

### Optimization 1: Fast Undistortion with LUT

```c
/**
 * @brief Optimized undistortion using integer LUT
 */
struct undistort_lut {
    int16_t *map_x;  /* Fixed-point coordinates */
    int16_t *map_y;
    uint8_t *interp_weights;  /* Interpolation weights */
};

static void init_fast_undistort_lut(
    struct camera_intrinsics *intrinsics,
    struct lens_distortion *distortion,
    unsigned int width,
    unsigned int height,
    struct undistort_lut *lut)
{
    lut->map_x = malloc(width * height * sizeof(int16_t));
    lut->map_y = malloc(width * height * sizeof(int16_t));
    lut->interp_weights = malloc(width * height * 4);
    
    for (unsigned int v = 0; v < height; v++) {
        for (unsigned int u = 0; u < width; u++) {
            unsigned int idx = v * width + u;
            
            /* Compute source coordinates */
            float x_norm = (u - intrinsics->cx) / intrinsics->fx;
            float y_norm = (v - intrinsics->cy) / intrinsics->fy;
            
            float x_undist, y_undist;
            remove_distortion(x_norm, y_norm, distortion,
                            &x_undist, &y_undist);
            
            float src_x = intrinsics->fx * x_undist + intrinsics->cx;
            float src_y = intrinsics->fy * y_undist + intrinsics->cy;
            
            /* Store as fixed-point */
            lut->map_x[idx] = (int16_t)(src_x * 16);  /* 4 fractional bits */
            lut->map_y[idx] = (int16_t)(src_y * 16);
            
            /* Precompute interpolation weights */
            uint8_t fx = (uint8_t)((src_x - (int)src_x) * 255);
            uint8_t fy = (uint8_t)((src_y - (int)src_y) * 255);
            
            lut->interp_weights[idx*4 + 0] = (255 - fx) * (255 - fy) / 255;
            lut->interp_weights[idx*4 + 1] = fx * (255 - fy) / 255;
            lut->interp_weights[idx*4 + 2] = (255 - fx) * fy / 255;
            lut->interp_weights[idx*4 + 3] = fx * fy / 255;
        }
    }
}

static void fast_undistort(
    uint8_t *src,
    uint8_t *dst,
    unsigned int width,
    unsigned int height,
    struct undistort_lut *lut)
{
    for (unsigned int i = 0; i < width * height; i++) {
        int src_x = lut->map_x[i] >> 4;
        int src_y = lut->map_y[i] >> 4;
        
        if (src_x < 0 || src_x >= width-1 ||
            src_y < 0 || src_y >= height-1) {
            dst[i] = 0;
            continue;
        }
        
        /* Fast bilinear using precomputed weights */
        uint8_t *w = &lut->interp_weights[i*4];
        
        unsigned int idx00 = src_y * width + src_x;
        unsigned int idx01 = idx00 + 1;
        unsigned int idx10 = idx00 + width;
        unsigned int idx11 = idx10 + 1;
        
        dst[i] = (src[idx00] * w[0] +
                 src[idx01] * w[1] +
                 src[idx10] * w[2] +
                 src[idx11] * w[3]) / 255;
    }
}
```

---

## 📝 Assessment Questions

### Conceptual Questions

1. **Explain the pinhole camera model and its limitations.**

2. **What causes radial and tangential distortion? How do they differ?**

3. **Why is Zhang's method effective for camera calibration?**

4. **Calculate the field of view for:**
   - Focal length: 800 pixels
   - Image width: 1920 pixels

5. **What is the minimum number of calibration images needed? Why?**

### Practical Challenges

1. **Implement fisheye lens distortion model.**

2. **Debug a calibration with RMS error > 2 pixels.**

3. **Optimize undistortion to run at 60fps for 1080p.**

4. **Design calibration pattern for wide-angle lens.**

---

## 📚 Further Reading & Resources

### Papers
- "A Flexible New Technique for Camera Calibration" - Zhang
- "Camera Calibration Toolbox for Matlab" - Bouguet

### Books
- "Multiple View Geometry" - Hartley & Zisserman
- "Computer Vision: Algorithms and Applications" - Szeliski

### Tools
- **OpenCV:** cv::calibrateCamera, cv::undistort
- **MATLAB Camera Calibration Toolbox**
- **Kalibr:** Multi-camera calibration

---

## 🎓 Summary

Today we covered:
- ✅ Pinhole camera model and projection equations
- ✅ Lens distortion models (radial and tangential)
- ✅ Zhang's calibration method
- ✅ Nonlinear refinement with bundle adjustment
- ✅ Undistortion map generation and application
- ✅ Complete calibration tool implementation
- ✅ Quality assessment and debugging techniques

**Key Takeaways:**
1. Calibration is essential for accurate 3D reconstruction
2. Multiple images with varied orientations improve quality
3. Distortion correction is critical for wide-angle lenses
4. Lookup tables enable real-time undistortion

**Next:** Day 12 - Advanced 3A Algorithms (Auto-Focus)

---

**Day 11 Complete** | Phase 3: Camera Systems & ISP | Week 2: Advanced Features
