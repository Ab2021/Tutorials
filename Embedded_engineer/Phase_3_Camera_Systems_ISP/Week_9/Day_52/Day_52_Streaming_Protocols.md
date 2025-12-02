# Day 52: Streaming Protocols (RTSP, WebRTC, HLS)
## Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming

---

## 🎯 Learning Objectives
1.  **Understand** the Streaming Landscape: Low Latency (WebRTC) vs Compatibility (RTSP) vs Scale (HLS).
2.  **Implement** an RTSP Server using GStreamer (`gst-rtsp-server`).
3.  **Analyze** the RTP/RTCP packet structure.
4.  **Deploy** a WebRTC stream for sub-second latency in the browser.
5.  **Configure** HLS (HTTP Live Streaming) for scaling to thousands of viewers.
6.  **Debug** streaming issues (Packet loss, Jitter, Firewall blocks).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera source, Network connection.
*   **Software:** GStreamer, VLC Player (Client), Browser.
*   **Knowledge:** UDP vs TCP, NAT Traversal (STUN/TURN).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: RTSP (Real-Time Streaming Protocol)
*   **Role:** The "Remote Control" for media. Handles Setup, Play, Pause, Teardown.
*   **Transport:** Usually RTP (Real-time Transport Protocol) over UDP.
*   **Latency:** Low (200ms - 2s).
*   **Use Case:** IP Cameras, VMS (Video Management Systems).
*   **Limitation:** Not natively supported in Web Browsers.

### 🔹 Part 2: WebRTC (Web Real-Time Communication)
*   **Role:** Peer-to-Peer audio/video/data.
*   **Transport:** SRTP (Secure RTP) over UDP.
*   **Latency:** Ultra Low (< 200ms).
*   **Use Case:** Video Conferencing (Zoom/Meet), Robot Control.
*   **Complexity:** Requires Signaling Server (SDP exchange) and ICE (NAT Traversal).

### 🔹 Part 3: HLS (HTTP Live Streaming) / DASH
*   **Role:** Chunk-based streaming over standard HTTP.
*   **Transport:** TS (Transport Stream) or fMP4 segments downloaded via HTTP GET.
*   **Latency:** High (5s - 30s).
*   **Use Case:** YouTube, Netflix, Broadcast.
*   **Advantage:** Works through firewalls, scales via CDN.

---

## 💻 Implementation Examples

### Example 1: RTSP Server (GStreamer)

Using `gst-rtsp-server` library.

```c
/**
 * @brief Simple RTSP Server
 * Stream available at rtsp://<ip>:8554/test
 */
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    GstRTSPServer *server = gst_rtsp_server_new();
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    
    // Pipeline: Videotestsrc -> H264Enc -> RTPPay
    gst_rtsp_media_factory_set_launch(factory, 
        "( videotestsrc ! x264enc tune=zerolatency ! rtph264pay name=pay0 pt=96 )");
        
    gst_rtsp_mount_points_add_factory(mounts, "/test", factory);
    
    gst_rtsp_server_attach(server, NULL);
    
    g_print("Stream ready at rtsp://127.0.0.1:8554/test\n");
    g_main_loop_run(loop);
    
    return 0;
}
```

### Example 2: WebRTC (Janus Gateway)

Instead of writing a C++ WebRTC stack from scratch (hard), we use a Gateway like Janus or MediaSoup.
1.  **Install Janus:** `apt install janus`.
2.  **Configure Streaming Plugin:** Edit `janus.plugin.streaming.jcfg`.
    ```ini
    [h264-sample]
    type = rtp
    id = 1
    description = H.264 Stream
    audio = no
    video = yes
    videoport = 8004
    videopt = 96
    videocodec = h264
    ```
3.  **Push RTP to Janus:**
    ```bash
    gst-launch-1.0 videotestsrc ! x264enc tune=zerolatency ! rtph264pay ! udpsink host=127.0.0.1 port=8004
    ```
4.  **View:** Open Janus Demo page in Browser.

### Example 3: HLS Segmentation

Creating an HLS playlist (`.m3u8`) and segments (`.ts`).

```bash
ffmpeg -i input.mp4 \
    -c:v libx264 -g 30 \
    -f hls \
    -hls_time 2 \
    -hls_list_size 5 \
    output.m3u8
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: RTSP Latency Test

**Objective:** Measure glass-to-glass latency.

**Steps:**
1.  Run RTSP Server (Example 1).
2.  Play with VLC (`vlc rtsp://...`). Latency ~2s (VLC buffers heavily).
3.  Play with `ffplay -fflags nobuffer rtsp://...`. Latency ~200ms.
4.  **Lesson:** The Client buffer settings matter as much as the Server.

### Lab 2: WebRTC vs RTSP

**Objective:** Compare user experience.

**Steps:**
1.  Set up Janus WebRTC (Example 2).
2.  Open Browser on Phone (on same WiFi).
3.  Wave hand.
4.  **Observation:** WebRTC feels instant. RTSP has a noticeable delay.

### Lab 3: Network Simulation (Packet Loss)

**Objective:** Observe artifacts.

**Steps:**
1.  Use `tc` (Traffic Control) on Linux to simulate 5% packet loss.
    ```bash
    sudo tc qdisc add dev eth0 root netem loss 5%
    ```
2.  Watch the stream.
3.  **RTSP (UDP):** Glitches, grey blocks, smearing.
4.  **RTSP (TCP):** Stuttering/Freezing (retransmission delay), but no visual corruption.

---

## 🐛 Debugging Techniques

### Debug 1: "Connection Refused"

**Symptom:** Client cannot connect.

**Cause:**
*   Firewall blocking port 8554 (RTSP) or 8004 (RTP).
*   Server bound to `127.0.0.1` (Localhost) instead of `0.0.0.0` (All interfaces).
*   **Fix:** Check `netstat -tulpn`.

### Debug 2: "No Video" (Black Screen)

**Symptom:** Connection established, but no video.

**Cause:**
*   RTP packets not flowing (UDP blocked).
*   Codec mismatch (Client doesn't support H.265).
*   Keyframe missing (Client waiting for next I-Frame).
*   **Fix:** Analyze with Wireshark. Look for RTP packets.

---

## ⚡ Performance Optimization

### Optimization 1: Hardware Packetization

*   Some HW encoders output pre-packetized RTP streams.
*   Avoids CPU overhead of `rtph264pay`.

### Optimization 2: Multicast

*   If streaming to 100 clients on LAN, Unicast (100 streams) kills bandwidth.
*   Use Multicast (1 stream, many listeners).
*   `udpsink host=224.1.1.1`.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does WebRTC need STUN/TURN servers?**
2.  **What is the function of the SDP (Session Description Protocol)?**
3.  **Why is TCP generally bad for live streaming?** (Head-of-Line Blocking).
4.  **How does HLS handle varying network speeds?** (Adaptive Bitrate Streaming - ABR).

### Practical Challenges

1.  **Build a "Baby Monitor":** Raspberry Pi Camera -> RTSP -> Phone App.
2.  **Create a "Video Doorbell":** Press button -> WebRTC Call to Browser.

---

## 📚 Further Reading & Resources

### Protocols
*   **RFC 3550:** RTP: A Transport Protocol for Real-Time Applications.
*   **RFC 2326:** Real Time Streaming Protocol (RTSP).

### Tools
*   **Wireshark:** Essential for debugging RTP/RTSP.
*   **Janus Gateway:** WebRTC Server.

---

## 🎓 Summary

Today we covered:
- ✅ **RTSP:** The standard for IP Cameras.
- ✅ **WebRTC:** The standard for Interaction.
- ✅ **HLS:** The standard for Broadcast.
- ✅ **GStreamer:** Building servers.
- ✅ **Network:** UDP vs TCP trade-offs.

**Next:** Day 53 - GStreamer Deep Dive (Pipelines, Plugins, Debugging).

---

**Day 52 Complete** | Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming
