# Day 75: Phase 3 Capstone Project - Implementation & Demo
## Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project

---

## 🎯 Learning Objectives
1.  **Implement** the "Intelligent Traffic Camera" software stack.
2.  **Configure** DeepStream SDK for LPR (License Plate Recognition).
3.  **Develop** the C++ Application to handle RTSP streaming and MQTT messaging.
4.  **Validate** the system under real-world conditions (Simulated).
5.  **Present** the final project: Demo, Architecture, and Lessons Learned.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Jetson Nano/Orin, Camera.
*   **Software:** NVIDIA JetPack 5.x/6.x, DeepStream 6.x.
*   **Models:** `LPRNet_usa_pruned.etlt` (from NVIDIA TAO).

---

## 💻 Implementation Guide

### 🔹 Step 1: DeepStream Configuration (`lpr_config.txt`)

Configure the GStreamer plugin `nvinfer` to use the LPR model.

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-file=LPRNet_usa_pruned.etlt
proto-file=LPRNet_usa.prototxt
labelfile-path=labels_lpr.txt
batch-size=1
process-mode=1
network-mode=1 # FP16
interval=0 # Process every frame
gie-unique-id=1
output-blob-names=tf_op_layer_Softmax
```

### 🔹 Step 2: The GStreamer Pipeline (C++)

Building the pipeline dynamically.

```cpp
#include <gst/gst.h>
#include <glib.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // 1. Create Elements
    GstElement *pipeline = gst_pipeline_new("traffic-cam");
    GstElement *source   = gst_element_factory_make("nvarguscamerasrc", "source");
    GstElement *streammux= gst_element_factory_make("nvstreammux", "mux");
    GstElement *pgie     = gst_element_factory_make("nvinfer", "primary-inference");
    GstElement *osd      = gst_element_factory_make("nvdsosd", "osd");
    GstElement *transform= gst_element_factory_make("nvvideoconvert", "transform");
    GstElement *encoder  = gst_element_factory_make("nvv4l2h264enc", "encoder");
    GstElement *rtsp     = gst_element_factory_make("rtspclientsink", "sink");

    // 2. Configure Elements
    g_object_set(G_OBJECT(pgie), "config-file-path", "lpr_config.txt", NULL);
    g_object_set(G_OBJECT(rtsp), "location", "rtsp://server/live", NULL);

    // 3. Link (Simplified)
    // source -> mux -> pgie -> osd -> transform -> encoder -> rtsp
    
    // 4. Start
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    // 5. Loop
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);
    
    return 0;
}
```

### 🔹 Step 3: Handling Metadata (Probe)

Extracting the License Plate string from the Inference metadata.

```cpp
GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (NvDsFrameMeta *frame_meta : batch_meta->frame_meta_list) {
        for (NvDsObjectMeta *obj_meta : frame_meta->obj_meta_list) {
            // Check if object is a License Plate
            if (obj_meta->class_id == CLASS_ID_LPR) {
                char *plate_number = obj_meta->text_params.display_text;
                g_print("Detected Plate: %s\n", plate_number);
                
                // Send to MQTT
                send_mqtt("traffic/plate", plate_number);
            }
        }
    }
    return GST_PAD_PROBE_OK;
}
```

---

## 🔬 Validation & Testing

### Test 1: Day Scenario
*   **Setup:** Point camera at a car (or monitor showing a car video) in bright light.
*   **Expected:** Plate detected with > 90% confidence. Bounding box drawn correctly.
*   **Pass/Fail:** Pass if text matches.

### Test 2: Night Scenario
*   **Setup:** Turn off lights. Use IR Illuminator (or simulate low light).
*   **Expected:** Image is B&W (IR mode). Plate is reflective (bright).
*   **Risk:** Headlight glare blinding the camera.
*   **Mitigation:** WDR (Wide Dynamic Range) enabled in ISP.

### Test 3: Stress Test
*   **Setup:** Run for 24 hours.
*   **Monitor:** Memory usage (Leak check), Temperature (Thermal throttling).
*   **Pass:** No crash, no freeze.

---

## 📢 Final Presentation Structure

### Slide 1: Title & Overview
*   "Intelligent Traffic Camera System"
*   Goal: Automated Toll Collection.

### Slide 2: Architecture
*   Diagram from Day 74.
*   Tech Stack: Jetson, DeepStream, MQTT.

### Slide 3: Challenges & Solutions
*   **Challenge:** Motion Blur at night.
*   **Solution:** Short exposure + High Gain + IR.
*   **Challenge:** Overheating.
*   **Solution:** Active cooling + FPS throttling.

### Slide 4: Demo Video
*   Show split screen: Live View vs Backend Log (MQTT messages).

### Slide 5: Future Improvements
*   Vehicle Classification (Car/Truck/Bus).
*   Speed Estimation (using Optical Flow).

---

## 🎓 Phase 3 Conclusion

Congratulations! You have completed **Phase 3: Camera Systems, SerDes & ISP Development**.

**You have mastered:**
*   **Physics:** Photons, Sensors, Optics.
*   **Hardware:** MIPI CSI-2, GMSL/FPD-Link, SerDes.
*   **Software:** V4L2 Drivers, ISP Tuning, GStreamer.
*   **AI:** Machine Vision, Deep Learning Integration.
*   **System:** Power, Thermal, Safety, Security.

**What's Next?**
**Phase 4: Embedded Linux & Kernel Development (Deep Dive).**
We will go deeper into the OS that powers these systems. Writing Board Support Packages (BSPs), Custom Drivers, and Yocto builds.

---

**Day 75 Complete** | Phase 3: Camera Systems & ISP | Week 13: Optimization & Final Project
