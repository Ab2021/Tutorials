# Frontend Architecture (2026 Standards)

The frontend is built with **Next.js 15 (App Router)** and React 19. It is designed to be lean, fast, and highly functional, offloading heavy processing to the browser while maintaining sub-second latency with the VPS.

## Core Pages

- `/` — Landing page.
- `/login`, `/register` — Authentication using NextAuth.js.
- `/dashboard` — List of past interviews.
- `/interview/[id]` — **The Live Interview Room** (Core functionality).
- `/report/[id]` — View the generated report post-interview.

## Live Interview Room UI Layout

```text
+---------------------------------------------------+
|  [End Interview]                     [Settings]   |
+---------------------------------------------------+
|   +--------------------+  +-------------------+   |
|   |  Candidate Video   |  |   AI Avatar /     |   |
|   |  (Webcam)          |  |   Audio Waveform  |   |
|   |  Engagement: 85%   |  |   [Speaking...]   |   |
|   +--------------------+  +-------------------+   |
|   +-------------------------------------------+   |
|   | Live Transcript                           |   |
|   +-------------------------------------------+   |
+---------------------------------------------------+
```

## Advanced Audio Capture Pipeline (2026)

Achieving low latency requires bypassing the main browser UI thread for audio processing.

1. **AudioWorklet:** We extract raw PCM data inside an `AudioWorkletNode`. This ensures that even if the React UI is busy animating waveforms, the audio capture never drops frames.
2. **WebRTC Noise Suppression:** Before transmission, the audio is routed through the browser's native WebRTC noise suppression API (enabled via `echoCancellation`, `noiseSuppression`, and `autoGainControl` constraints in `getUserMedia`). This prevents background noise from degrading the STT engine on the server.
3. **Client-Side VAD (Voice Activity Detection):** Using `onnxruntime-web`, a lightweight Silero VAD model runs in the browser. It acts as a gatekeeper: audio chunks are ONLY transmitted over the network when the candidate is actively speaking.

## Networking Transition: WebSockets to WebTransport

While the MVP starts with WebSockets (WSS), the architecture is designed to transition to **WebTransport** (HTTP/3).
- **The Problem:** WebSockets over TCP suffer from Head-of-Line blocking. If a packet drops, audio stutters.
- **The Solution:** WebTransport streams over UDP (QUIC). This allows the frontend to stream PCM audio unreliably (where dropped packets are simply ignored rather than stalling the stream), drastically improving audio latency and smoothness on poor network connections, without the massive setup complexity of a full WebRTC peer connection.

## Client-Side Video Analysis (WebGPU)

Using `@mediapipe/tasks-vision` or **LiteRT.js**, the frontend tracks 468 facial landmarks.
- **Delegate:** Must be set to `GPU` (WebGPU) to maintain 30 FPS without destroying laptop battery life.
- **Telemetry:** Calculates engagement and emotion locally, sending only a 1KB JSON payload over the WebSocket every 500ms to the backend. Raw video is NEVER transmitted.
