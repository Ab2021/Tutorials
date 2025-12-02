# Day 53: GStreamer Deep Dive
## Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming

---

## 🎯 Learning Objectives
1.  **Master** the GStreamer Architecture: Elements, Pads, Bins, Pipelines, Bus.
2.  **Construct** complex pipelines using `gst-launch-1.0`.
3.  **Develop** GStreamer applications in C/C++.
4.  **Debug** pipelines using Graphviz (`GST_DEBUG_DUMP_DOT_DIR`) and debug logs.
5.  **Write** a simple GStreamer Plugin (Transform Element).
6.  **Handle** dynamic pipelines (Pad Added signals).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** PC or Embedded Board.
*   **Software:** GStreamer Development Libraries (`libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`).
*   **Knowledge:** C Programming, GObject (Basic concept).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Core Concepts
*   **Element:** The basic building block (Source, Filter, Sink).
*   **Pad:** The input/output port of an element.
    *   **Source Pad:** Produces data.
    *   **Sink Pad:** Consumes data.
    *   **Caps (Capabilities):** Describes the data format (e.g., `video/x-raw, width=1920`).
*   **Bin:** A container for elements (e.g., `pipeline` is a top-level bin).
*   **Bus:** Message passing system (Errors, EOS, Tags) from pipeline to application.

### 🔹 Part 2: Pipeline States
*   **NULL:** Default state. No resources allocated.
*   **READY:** Resources allocated, but not streaming.
*   **PAUSED:** Streaming, but clock is stopped (Preroll).
*   **PLAYING:** Streaming and clock running.

### 🔹 Part 3: Buffers and Events
*   **Buffer:** Contains the actual media data (pixels, audio samples) + Timestamp (PTS/DTS).
*   **Event:** Control signals (EOS, Seek, Flush, QOS). Flow upstream or downstream.

---

## 💻 Implementation Examples

### Example 1: Complex `gst-launch`

Split a stream: Display one copy, Save another.

```bash
gst-launch-1.0 -v \
    videotestsrc ! video/x-raw,width=640,height=480 ! tee name=t \
    t. ! queue ! videoconvert ! autovideosink \
    t. ! queue ! x264enc ! mp4mux ! filesink location=test.mp4
```

*   **tee:** Splits data 1-to-N.
*   **queue:** Creates a new thread. Essential after `tee` to prevent one branch from blocking the other.

### Example 2: Dynamic Pipeline (C++)

Handling `rtspsrc` which creates pads dynamically (sometimes audio, sometimes video).

```cpp
/**
 * @brief Handling "pad-added" signal
 */
#include <gst/gst.h>

void on_pad_added(GstElement *src, GstPad *new_pad, gpointer data) {
    GstElement *sink = (GstElement *)data;
    GstPad *sink_pad = gst_element_get_static_pad(sink, "sink");
    
    // Check if already linked
    if (gst_pad_is_linked(sink_pad)) {
        g_object_unref(sink_pad);
        return;
    }
    
    // Check Caps (Is it video?)
    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    
    if (g_str_has_prefix(name, "video/x-raw")) {
        gst_pad_link(new_pad, sink_pad);
        g_print("Linked video pad.\n");
    }
    
    gst_caps_unref(caps);
    g_object_unref(sink_pad);
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    
    GstElement *pipeline = gst_pipeline_new("test-pipeline");
    GstElement *source = gst_element_factory_make("uridecodebin", "source");
    GstElement *sink = gst_element_factory_make("autovideosink", "sink");
    
    g_object_set(source, "uri", "https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm", NULL);
    
    gst_bin_add_many(GST_BIN(pipeline), source, sink, NULL);
    
    // Listen for new pads
    g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), sink);
    
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    // Main Loop...
}
```

### Example 3: Writing a Plugin (Boilerplate)

Creating a "MyFilter" element.

```c
/* gstmyfilter.c */
#include "gstmyfilter.h"

G_DEFINE_TYPE (GstMyFilter, gst_my_filter, GST_TYPE_BASE_TRANSFORM);

static GstFlowReturn gst_my_filter_transform_ip (GstBaseTransform *base, GstBuffer *buf) {
    GstMyFilter *filter = GST_MY_FILTER (base);
    
    // Map Buffer (Read/Write)
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_READWRITE);
    
    // Invert Colors (Simple processing)
    for (int i = 0; i < map.size; i++) {
        map.data[i] = 255 - map.data[i];
    }
    
    gst_buffer_unmap(buf, &map);
    return GST_FLOW_OK;
}

static void gst_my_filter_class_init (GstMyFilterClass *klass) {
    GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);
    base_transform_class->transform_ip = gst_my_filter_transform_ip;
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Visualizing the Pipeline

**Objective:** See what's happening inside.

**Steps:**
1.  Set environment: `export GST_DEBUG_DUMP_DOT_DIR=/tmp`.
2.  Run any pipeline.
3.  Convert dot file: `dot -Tpng /tmp/*.dot > pipeline.png`.
4.  **Result:** A graph showing all elements, pads, and caps negotiation results.

### Lab 2: Performance Profiling

**Objective:** Find bottlenecks.

**Steps:**
1.  Use `gst-launch-1.0 -v ...`.
2.  Look for "QoS" messages (Quality of Service). If the sink says "dropped", the pipeline is too slow.
3.  Add `fpsdisplaysink` instead of `autovideosink`.
    ```bash
    gst-launch-1.0 videotestsrc ! fpsdisplaysink
    ```

### Lab 3: Zero-Copy Pipeline

**Objective:** Efficient memory usage.

**Steps:**
1.  Use `v4l2src` (DMA Buf) -> `v4l2h264enc` (DMA Buf).
2.  Verify "memory:DMABuf" in caps.
3.  **Goal:** CPU usage should be < 5%.

---

## 🐛 Debugging Techniques

### Debug 1: "Internal Data Stream Error"

**Symptom:** Pipeline stops immediately.

**Cause:**
*   Caps mismatch. Element A outputs YUV, Element B expects RGB.
*   **Fix:** Add `videoconvert` between them.
*   **Fix:** Check logs: `GST_DEBUG=3 gst-launch-1.0 ...`

### Debug 2: "Not Negotiated"

**Symptom:** Pipeline fails to link.

**Cause:**
*   Incompatible formats.
*   **Fix:** Check the "Pad Templates" of the elements using `gst-inspect-1.0 <element_name>`.

---

## ⚡ Performance Optimization

### Optimization 1: Queues

*   GStreamer is single-threaded by default (mostly).
*   Insert `queue` elements to create thread boundaries.
*   Example: `source ! queue ! process ! queue ! sink`.
*   Allows Source, Process, and Sink to run on different CPU cores.

### Optimization 2: Typefinding

*   `decodebin` spends time guessing the format.
*   If you know the format (e.g., H.264), use specific parsers (`h264parse`) instead of generic decoders.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the difference between `videoconvert` and `videoscale`?**
2.  **Why do we need `h264parse` before `avdec_h264`?** (To frame the byte stream into NAL units).
3.  **What happens when a buffer has no timestamp?**
4.  **What is "Caps Negotiation"?**

### Practical Challenges

1.  **Build a "Mosaic" Pipeline:** Combine 4 video streams into a 2x2 grid using `videomixer` or `compositor`.
2.  **Implement "Motion Detection":** Use `gdp` (GStreamer Data Protocol) or write an element that analyzes motion vectors.

---

## 📚 Further Reading & Resources

### Documentation
*   **GStreamer Application Development Manual.**
*   **Plugin Writer's Guide.**

### Tools
*   **gst-inspect-1.0:** List element details.
*   **GstShark:** Profiling tool (CPU, Latency, Framerate).

---

## 🎓 Summary

Today we covered:
- ✅ **Architecture:** Elements, Pads, Bins.
- ✅ **Tools:** `gst-launch`, `gst-inspect`.
- ✅ **Coding:** C API and Signals.
- ✅ **Debugging:** DOT graphs and Logs.
- ✅ **Plugins:** Writing custom elements.

**Next:** Day 54 - WebRTC Deep Dive (Signaling, ICE, STUN/TURN).

---

**Day 53 Complete** | Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming
