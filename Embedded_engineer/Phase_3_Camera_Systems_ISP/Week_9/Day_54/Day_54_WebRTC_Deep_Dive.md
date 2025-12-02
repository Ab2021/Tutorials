# Day 54: WebRTC Deep Dive (Signaling, ICE, STUN/TURN)
## Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming

---

## 🎯 Learning Objectives
1.  **Understand** the WebRTC Connection Flow: Signaling -> ICE Candidates -> DTLS Handshake -> Media.
2.  **Implement** a Signaling Server using WebSockets (Node.js/Python).
3.  **Configure** STUN and TURN servers for NAT Traversal.
4.  **Exchange** SDP (Session Description Protocol) Offer and Answer.
5.  **Use** Data Channels for low-latency telemetry (Robot Control).
6.  **Debug** connection failures using `chrome://webrtc-internals`.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Two devices on different networks (e.g., PC and Phone on 4G).
*   **Software:** Node.js, `coturn` (TURN server).
*   **Knowledge:** JavaScript Promises, JSON.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Connection Problem
*   **NAT (Network Address Translation):** Most devices don't have a public IP. They are hidden behind a router (192.168.x.x).
*   **Peer-to-Peer:** To connect Device A to Device B, they need to know each other's *Public* IP and Port.

### 🔹 Part 2: ICE (Interactive Connectivity Establishment)
*   **STUN (Session Traversal Utilities for NAT):** "Who am I?" Device asks STUN server: "What is my Public IP?". Server replies: "You are 203.0.113.5:4500".
*   **TURN (Traversal Using Relays around NAT):** If P2P fails (Symmetric NAT), relay all data through a TURN server. Expensive (bandwidth) but guaranteed to work.
*   **Candidates:**
    *   **Host:** Local IP (192.168.1.5).
    *   **Srflx:** Server Reflexive (Public IP via STUN).
    *   **Relay:** TURN Server IP.

### 🔹 Part 3: Signaling
*   WebRTC does NOT define how peers find each other. You must build a "Signaling Channel" (usually WebSockets).
*   **SDP (Session Description Protocol):** Describes the media (Codec, Resolution, Encryption).
*   **Flow:**
    1.  Peer A creates **Offer** (SDP). Sends to B via Signaling.
    2.  Peer B sets Remote Desc. Creates **Answer** (SDP). Sends to A.
    3.  Both exchange **ICE Candidates** (IP:Port pairs) as they are discovered.

---

## 💻 Implementation Examples

### Example 1: Simple Signaling Server (Node.js)

```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function connection(ws) {
  ws.on('message', function incoming(message) {
    // Broadcast to everyone else (Naive implementation)
    wss.clients.forEach(function each(client) {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  });
});
```

### Example 2: WebRTC Client (JavaScript)

```javascript
const config = {
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
};
const pc = new RTCPeerConnection(config);
const ws = new WebSocket('ws://localhost:8080');

// 1. Handle ICE Candidates
pc.onicecandidate = event => {
    if (event.candidate) {
        ws.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
    }
};

// 2. Handle Incoming Stream
pc.ontrack = event => {
    document.getElementById('remoteVideo').srcObject = event.streams[0];
};

// 3. Create Offer (Initiator)
async function startCall() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    stream.getTracks().forEach(track => pc.addTrack(track, stream));
    
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: 'offer', sdp: offer }));
}

// 4. Handle Signaling Messages
ws.onmessage = async message => {
    const data = JSON.parse(message.data);
    if (data.type === 'offer') {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        ws.send(JSON.stringify({ type: 'answer', sdp: answer }));
    } else if (data.type === 'answer') {
        await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
    } else if (data.type === 'candidate') {
        await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
    }
};
```

### Example 3: Data Channels (Robot Control)

Sending JSON commands with low latency.

```javascript
// Sender
const dc = pc.createDataChannel("robot_control");
dc.onopen = () => dc.send(JSON.stringify({ cmd: "move_forward", speed: 50 }));

// Receiver
pc.ondatachannel = event => {
    const receiveChannel = event.channel;
    receiveChannel.onmessage = e => {
        const cmd = JSON.parse(e.data);
        console.log("Received command:", cmd);
    };
};
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: STUN vs TURN

**Objective:** Force Relay.

**Steps:**
1.  Use `stun:stun.l.google.com:19302`. Connect two tabs.
    *   Result: `srflx` or `host` candidates used. Low latency.
2.  Block UDP ports on your firewall.
3.  Connection fails.
4.  Configure a TURN server (e.g., `turn:my-turn-server.com`).
5.  **Result:** Connection succeeds via `relay` candidate. Latency increases slightly.

### Lab 2: Latency Measurement

**Objective:** Measure Data Channel latency.

**Steps:**
1.  Send `timestamp` from A to B.
2.  B echoes it back to A.
3.  A calculates `RTT = Now - timestamp`.
4.  **Goal:** < 100ms RTT on 4G.

### Lab 3: Bandwidth Estimation

**Objective:** Observe Adaptive Bitrate.

**Steps:**
1.  Open `chrome://webrtc-internals`.
2.  Look at `bweCompound` (Bandwidth Estimation).
3.  Simulate network congestion (throttle via DevTools).
4.  **Observation:** WebRTC automatically lowers resolution/framerate to maintain low latency.

---

## 🐛 Debugging Techniques

### Debug 1: "ICE Connection Failed"

**Symptom:** Signaling works, but video never starts.

**Cause:**
*   Symmetric NAT (Corporate Firewall).
*   UDP blocked.
*   No TURN server configured.
*   **Fix:** Check `chrome://webrtc-internals` -> ICE Candidates. If you only see `host` candidates, STUN failed.

### Debug 2: One-Way Audio/Video

**Symptom:** A sees B, but B can't see A.

**Cause:**
*   Firewall allows outgoing UDP but blocks incoming.
*   Codec mismatch (A sends H.264, B only supports VP8).
*   **Fix:** Check SDP "m=video" lines to verify common codecs.

---

## ⚡ Performance Optimization

### Optimization 1: Codec Selection

*   **VP8:** Default. Good CPU usage.
*   **H.264:** Hardware accelerated on many mobiles. Good for battery.
*   **AV1:** Best quality, high CPU. Avoid for mobile unless HW support exists.

### Optimization 2: Simulcast

*   Send 3 streams (High, Medium, Low quality) simultaneously.
*   SFU (Selective Forwarding Unit) server forwards the appropriate stream to each client based on their bandwidth.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the difference between an SFU (Selective Forwarding Unit) and an MCU (Multipoint Control Unit)?**
2.  **Why is TCP (WebSockets) used for Signaling but UDP (RTP) for Media?**
3.  **What information is contained in an ICE Candidate?**
4.  **How does DTLS ensure security?**

### Practical Challenges

1.  **Build a "File Transfer" App:** Use Data Channels to send a large file (chunked) between two browsers P2P.
2.  **Implement "Screen Sharing":** Use `getDisplayMedia` API instead of `getUserMedia`.

---

## 📚 Further Reading & Resources

### Standards
*   **W3C WebRTC 1.0 Specification.**

### Tools
*   **coturn:** Open source TURN server.
*   **PeerJS:** Library that abstracts the complexity.

---

## 🎓 Summary

Today we covered:
- ✅ **Signaling:** The missing piece.
- ✅ **ICE:** STUN/TURN for NAT traversal.
- ✅ **SDP:** Negotiating capabilities.
- ✅ **Data Channels:** For non-media data.
- ✅ **Debugging:** `webrtc-internals`.

**Next:** Day 55 - Week 9 Review & Streaming Project.

---

**Day 54 Complete** | Phase 3: Camera Systems & ISP | Week 9: Video Encoding & Streaming
