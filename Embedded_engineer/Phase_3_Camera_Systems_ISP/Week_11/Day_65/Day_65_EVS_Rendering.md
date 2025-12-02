# Day 65: EVS Display & Rendering
## Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS

---

## 🎯 Learning Objectives
1.  **Understand** the EVS Display Interface (`IEvsDisplay`).
2.  **Implement** the Rendering Pipeline using OpenGL ES 2.0/3.0.
3.  **Import** External Textures (`GL_TEXTURE_EXTERNAL_OES`) from Camera buffers.
4.  **Draw** Overlays (Guidelines, Text) on top of the video.
5.  **Manage** Display State (Exclusive access vs Shared).
6.  **Debug** Rendering artifacts (Tearing, Stuttering).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Display connected to the board.
*   **Software:** EVS App source code.
*   **Knowledge:** OpenGL ES Shaders (Vertex/Fragment), EGL.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Display HAL (`IEvsDisplay`)
*   Provides an abstraction over the physical display (usually via DRM/KMS).
*   **Methods:**
    *   `getDisplayInfo()`: Resolution, DPI.
    *   `getTargetBuffer()`: Get a buffer to draw into.
    *   `returnTargetBufferForDisplay()`: Post the buffer to the screen (Page Flip).

### 🔹 Part 2: Zero-Copy Rendering (OES Textures)
*   Standard GL textures (`GL_TEXTURE_2D`) require RGB data.
*   Camera buffers are often YUV (NV21/YUYV).
*   **OES Extension:** `GL_TEXTURE_EXTERNAL_OES` allows the GPU to read YUV data directly and convert it to RGB in the shader hardware.
*   **EGLImage:** The bridge between the `AHardwareBuffer` (Gralloc) and the GL Texture.

### 🔹 Part 3: Overlay Graphics
*   Rear View Cameras need **Dynamic Guidelines** (trajectory lines based on Steering Angle).
*   These are drawn as geometry (lines/triangles) *over* the video texture in a second render pass.

---

## 💻 Implementation Examples

### Example 1: Fragment Shader for OES Texture

The shader that draws the camera frame.

```glsl
#extension GL_OES_EGL_image_external : require
precision mediump float;

varying vec2 vTextureCoord;
uniform samplerExternalOES sTexture; // The Camera Buffer

void main() {
    // Sample the YUV buffer, GPU converts to RGB automatically
    gl_FragColor = texture2D(sTexture, vTextureCoord);
}
```

### Example 2: Importing Buffer to GL (EGL)

C++ code to bind the camera buffer.

```cpp
void bind_camera_frame(BufferDesc& buf) {
    // 1. Create EGL Image from Native Buffer
    EGLClientBuffer cbuf = (EGLClientBuffer)buf.memHandle;
    EGLint attrs[] = { EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE };
    
    EGLImageKHR image = eglCreateImageKHR(display, EGL_NO_CONTEXT, 
                                          EGL_NATIVE_BUFFER_ANDROID, 
                                          cbuf, attrs);
                                          
    // 2. Bind to Texture
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, textureId);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, image);
    
    // 3. Cleanup (Image can be destroyed after binding)
    eglDestroyImageKHR(display, image);
}
```

### Example 3: Drawing Guidelines

Updating vertex buffer based on Steering Angle.

```cpp
void draw_guidelines(float steering_angle) {
    // Calculate curve points based on Ackermann steering geometry
    std::vector<float> vertices = calculate_trajectory(steering_angle);
    
    glUseProgram(lineShader);
    glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, vertices.data());
    glEnableVertexAttribArray(posLoc);
    
    // Draw Yellow Lines
    glUniform4f(colorLoc, 1.0, 1.0, 0.0, 1.0); 
    glDrawArrays(GL_LINE_STRIP, 0, vertices.size() / 2);
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Basic Rendering

**Objective:** Get an image on screen.

**Steps:**
1.  Run `evs_app`.
2.  Verify the camera image fills the screen.
3.  **Experiment:** Modify the Vertex Shader to scale the image by 0.5x.
    ```glsl
    gl_Position = vec4(aPosition * 0.5, 0.0, 1.0);
    ```
4.  **Result:** Video should appear in a small window in the center.

### Lab 2: Adding a Static Overlay

**Objective:** Draw a red box.

**Steps:**
1.  Create a second Shader Program (Simple Color).
2.  Define vertices for a rectangle.
3.  Draw it *after* drawing the video texture (Painter's Algorithm).
4.  **Result:** Red box on top of video.

### Lab 3: Dynamic Guidelines

**Objective:** Animate the overlay.

**Steps:**
1.  Subscribe to `PERF_VEHICLE_STEERING_ANGLE` in VHAL.
2.  Pass the angle to the render loop.
3.  Rotate the guidelines based on the angle.
4.  **Observation:** Lines bend as you "turn the wheel" (inject VHAL events).

---

## 🐛 Debugging Techniques

### Debug 1: "Black Screen"

**Symptom:** App runs, no errors, but screen is black.

**Cause:**
*   Z-Order issue. The EVS layer is behind the Black background layer.
*   Shader compilation failure (Check `glGetShaderInfoLog`).
*   Texture coordinates are all 0.
*   **Fix:** Check `adb shell dumpsys SurfaceFlinger` to see layer composition.

### Debug 2: Tearing

**Symptom:** Horizontal lines break the image during motion.

**Cause:**
*   Double Buffering not working. We are drawing to the Front Buffer.
*   **Fix:** Ensure `eglSwapBuffers` or `returnTargetBufferForDisplay` is waiting for VSYNC.

---

## ⚡ Performance Optimization

### Optimization 1: Mesh Distortion Correction

*   Instead of drawing a simple quad (2 triangles), draw a **Mesh** (grid of triangles).
*   Distort the mesh vertices in the Vertex Shader to correct for Lens Distortion (Fisheye).
*   **Benefit:** Much faster than doing it in CPU or ISP. Zero cost on modern GPUs.

### Optimization 2: Alpha Blending

*   Overlays should be semi-transparent.
*   `glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);`
*   **Cost:** Blending is expensive (Read-Modify-Write). Minimize the area of overlays.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is `GL_TEXTURE_EXTERNAL_OES`?** (A texture target for YUV/External buffers).
2.  **Why do we use EGL?** (To interface OpenGL with the native window system).
3.  **How does "Fisheye Correction" work in a Vertex Shader?** (Move vertices to counteract the lens distortion).
4.  **What is "VSYNC"?** (Vertical Synchronization - prevents tearing).

### Practical Challenges

1.  **Implement "Picture-in-Picture":** Draw the Rear Camera full screen, and a second camera (e.g., Trailer) in a small box in the corner.
2.  **Create a "Distance Warning":** Change the color of the guidelines from Green to Red as the Ultrasonic Sensor distance (from VHAL) decreases.

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenGL ES 3.0 Reference.**
*   **Android EGL Specification.**

---

## 🎓 Summary

Today we covered:
- ✅ **Rendering:** OpenGL ES pipeline.
- ✅ **Textures:** Importing YUV via OES.
- ✅ **Overlays:** Drawing guidelines.
- ✅ **EGL:** The glue between Buffer and GPU.
- ✅ **Distortion:** GPU-based correction.

**Next:** Day 66 - EVS Surround View (Stitching & Multi-Camera).

---

**Day 65 Complete** | Phase 3: Camera Systems & ISP | Week 11: Android Automotive & EVS
