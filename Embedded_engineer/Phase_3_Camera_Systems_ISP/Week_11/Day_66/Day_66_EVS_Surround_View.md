# Day 66: EVS Surround View (AVM)
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Understand** the concept of Around View Monitor (AVM) / Surround View.
2.  **Implement** the 4-Camera Capture pipeline in EVS.
3.  **Perform** Image Stitching (Homography) on the GPU.
4.  **Calibrate** the Extrinsic parameters (Camera Pose) relative to the car.
5.  **Render** the "Bowl View" (3D projection).
6.  **Optimize** bandwidth for 4 simultaneous HD streams.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** 4 Fisheye Cameras (Front, Rear, Left, Right).
*   **Software:** EVS App with Multi-Camera support.
*   **Knowledge:** Homography, Projection Matrices, UV Mapping.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The AVM Concept
*   **Goal:** Create a virtual "Bird's Eye View" of the car.
*   **Input:** 4 ultra-wide (180°+) fisheye cameras.
*   **Process:**
    1.  **Undistort:** Remove fisheye distortion.
    2.  **Project:** Project images onto a ground plane (IPM - Inverse Perspective Mapping).
    3.  **Stitch:** Blend overlapping regions (Alpha Blending).
    4.  **Render:** Draw the car model in the center.

### 🔹 Part 2: Calibration (Extrinsic)
*   We need to know exactly where each camera is mounted ($x, y, z$) and its rotation (Pitch, Yaw, Roll).
*   **Calibration Pattern:** Large checkerboard mats placed on the ground around the car.
*   **Result:** A Homography Matrix ($H$) for each camera that maps pixel coordinates $(u, v)$ to ground coordinates $(X, Y)$.

### 🔹 Part 3: The "Bowl" Model
*   Flat IPM stretches pixels too much at the horizon.
*   **Bowl View:** Projects images onto a 3D bowl shape.
    *   **Ground:** Flat plane.
    *   **Horizon:** Curved up.
    *   **Sky:** Not shown.
*   Provides a more natural look.

---

## 💻 Implementation Examples

### Example 1: Multi-Camera EVS App

Opening 4 cameras.

```cpp
struct CameraStream {
    sp<IEvsCamera> hw;
    GLuint textureId;
};

CameraStream cams[4];
const char* camIds[] = {"front", "rear", "left", "right"};

void start_surround_view() {
    for (int i=0; i<4; i++) {
        cams[i].hw = pEnum->openCamera(camIds[i], ...);
        cams[i].hw->startVideoStream(new StreamHandler(i));
    }
}
```

### Example 2: GPU Stitching (Fragment Shader)

Blending 4 textures based on a "Mask" or Look-Up Table (LUT).

```glsl
uniform samplerExternalOES uFront;
uniform samplerExternalOES uRear;
uniform samplerExternalOES uLeft;
uniform samplerExternalOES uRight;
uniform sampler2D uBlendMask; // R=Front, G=Rear, B=Left, A=Right

varying vec2 vTexCoord;

void main() {
    vec4 mask = texture2D(uBlendMask, vTexCoord);
    
    vec4 cFront = texture2D(uFront, map_to_front(vTexCoord));
    vec4 cRear  = texture2D(uRear,  map_to_rear(vTexCoord));
    vec4 cLeft  = texture2D(uLeft,  map_to_left(vTexCoord));
    vec4 cRight = texture2D(uRight, map_to_right(vTexCoord));
    
    // Weighted Blend
    gl_FragColor = cFront*mask.r + cRear*mask.g + cLeft*mask.b + cRight*mask.a;
}
```

*   **Note:** In reality, we use a **Mesh-based approach** (Vertex Shader) for efficiency, where the UV coordinates of the mesh vertices are pre-calculated to point to the correct camera texture.

### Example 3: Mesh Generation (LUT)

Generating the 3D Bowl Mesh.

```cpp
struct Vertex {
    float x, y, z;       // Screen Position
    float u, v;          // Texture Coordinate
    float blend_weight;  // Alpha for blending
    int texture_index;   // Which camera?
};

std::vector<Vertex> generate_bowl_mesh() {
    // Iterate grid
    for (int i=0; i<ROWS; i++) {
        for (int j=0; j<COLS; j++) {
            // Calculate 3D position on Bowl
            // Project to Camera Image Plane
            // Store UVs
        }
    }
    return mesh;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: 4-Stream Bandwidth Test

**Objective:** Can the bus handle it?

**Steps:**
1.  Open 4 cameras at 1280x720 @ 30fps.
2.  Total Bandwidth: $1280 \times 720 \times 2 (YUYV) \times 30 \times 4 \approx 220$ MB/s.
3.  **Monitor:** Check CSI-2 bus utilization and Memory Bandwidth.
4.  **Observation:** If frames drop, reduce resolution or frame rate.

### Lab 2: Manual Stitching

**Objective:** Align two cameras.

**Steps:**
1.  Park car on a line.
2.  Display Front and Left cameras side-by-side.
3.  Adjust the "Shift" (X/Y offset) in the shader until the line is continuous across the boundary.
4.  **Result:** A rough manual calibration.

### Lab 3: The "Ghost" Car

**Objective:** Render the car model.

**Steps:**
1.  Load a `.obj` model of the car (top-down view).
2.  Render it in the center of the screen (on top of the stitched video).
3.  **Result:** Hides the "Blind Spot" under the car where no cameras can see.

---

## 🐛 Debugging Techniques

### Debug 1: "Seams" in the Image

**Symptom:** Lines on the ground appear broken or doubled at the stitching boundary.

**Cause:**
*   Calibration error. The Homography is valid for the ground plane ($Z=0$). Objects above ground (e.g., another car) will always have seams (Parallax Error).
*   **Fix:** Perfect calibration helps, but Parallax is physically unavoidable without true 3D reconstruction.

### Debug 2: Exposure Differences

**Symptom:** Front camera is bright (Sun), Rear is dark (Shadow). The stitch looks ugly.

**Cause:**
*   Independent Auto-Exposure (AE) on each camera.
*   **Fix:** Implement "Master AE". Calculate average brightness of all 4 images, and force the same Exposure/Gain on all 4 cameras.

---

## ⚡ Performance Optimization

### Optimization 1: Offline LUT Generation

*   Do NOT calculate mappings in the Shader.
*   Calculate the Mesh (XYZ -> UV) **offline** (on PC) or once at startup.
*   The Vertex Shader just passes the pre-calculated UVs to the Fragment Shader.

### Optimization 2: VBO (Vertex Buffer Object)

*   Upload the Mesh to GPU memory (VBO) once.
*   Draw call becomes extremely cheap.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Inverse Perspective Mapping" (IPM)?** (Removing perspective effect to make parallel lines parallel).
2.  **Why do we need Fisheye lenses for AVM?** (To see the corners and wheels with only 4 cameras).
3.  **What is the "Parallax Problem" in stitching?**
4.  **Why is "Master AE" necessary?**

### Practical Challenges

1.  **Implement "Wheel View":** Create a view mode that zooms in on the front-right wheel (using the Right camera) to help parking near a curb.
2.  **Auto-Calibration:** Use feature matching (ORB) in the overlap regions to automatically align the cameras while driving.

---

## 📚 Further Reading & Resources

### Papers
*   **"Surround View Camera System for ADAS"** (Various IEEE papers).

### Tools
*   **OpenCV `calib3d`:** For finding Homography.

---

## 🎓 Summary

Today we covered:
- ✅ **AVM:** The ultimate parking aid.
- ✅ **Pipeline:** 4 Cams -> Stitch -> Render.
- ✅ **Calibration:** Aligning the world.
- ✅ **Blending:** Hiding the seams.
- ✅ **Optimization:** LUT-based rendering.

**Next:** Day 67 - Week 11 Review & EVS Project.

---

**Day 66 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
