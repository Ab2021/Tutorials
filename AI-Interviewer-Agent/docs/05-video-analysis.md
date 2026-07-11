# Video Analysis Architectural Analysis

This document explores the architectural reasoning behind the video analysis pipeline of the AI Interviewer Agent. The goal of this component is to read the candidate's facial expressions, engagement levels, and eye contact to provide post-interview analytics. Crucially, this must be accomplished without overwhelming the constrained Hostinger VPS (2 vCPU, 8GB RAM) and without violating strict privacy standards.

## 1. The Client-Side Execution Mandate

The most significant architectural decision in the video pipeline is the absolute prohibition of server-side video processing. 

### 1.1 The Problems with Server-Side Processing
If the architecture attempted to stream raw video frames (e.g., via WebRTC or WebSocket binary frames) to the VPS for processing:
1. **Bandwidth Collapse:** Streaming 720p video at 30 FPS requires megabits of continuous bandwidth per user. The Hostinger VPS network interface would saturate almost immediately with just a few concurrent users.
2. **CPU Exhaustion:** Processing computer vision models (like face landmark detection) on a CPU is massively expensive. A single instance would consume the entire 2 vCPU budget, leaving nothing for the STT, TTS, or Web server.
3. **Privacy Liability:** Transmitting and storing raw biometric video data in the cloud introduces severe GDPR, CCPA, and general privacy liabilities. Securing this data in transit and at rest is a massive undertaking.

### 1.2 The Client-Side Solution
By leveraging modern web APIs, the entire computer vision pipeline is pushed down to the "Ultra-Edge"—the candidate's own browser. 
The browser accesses the webcam, runs the neural network locally, extracts lightweight metadata (JSON objects containing scores and coordinates), and only transmits this metadata to the server.

- *Pros:* Zero server CPU usage for video processing. Minimal bandwidth usage (sending a 1KB JSON payload twice a second). Total privacy preservation, as raw pixels never leave the user's device.
- *Cons:* Relies heavily on the computational power of the candidate's device. A user on a five-year-old budget smartphone may experience severe battery drain or browser tab freezing.

## 2. Selected Framework: LiteRT.js (WebGPU)

To execute neural networks in the browser at high frame rates without destroying battery life, the architecture relies on **LiteRT.js** (formerly TFLite for Web) utilizing the **WebGPU** backend, often orchestrated via the MediaPipe Tasks API.

### 2.1 The Power of WebGPU
WebGPU is the modern successor to WebGL. It provides low-overhead access to the device's underlying graphics hardware (GPU). 
- *Pros:* When executing models like the MediaPipe FaceMesh (which tracks 468 facial landmarks), WebGPU can process frames in milliseconds, maintaining a locked 30 FPS. It is vastly more power-efficient than CPU-bound WebAssembly (WASM).
- *Cons:* Browser support, while growing rapidly in 2026, is not absolute. Older browsers or specific mobile operating systems may lack WebGPU implementation.

### 2.2 Fallback Mechanisms (WASM)
Because WebGPU is not guaranteed, the architecture mandates a silent, automatic fallback to WebAssembly (WASM) executed on the client's CPU. 
If WebGPU initialization fails, the application catches the error and re-initializes the vision model with the CPU delegate. To prevent this from freezing the browser UI thread, the polling rate is aggressively throttled. Instead of processing every frame (30 FPS), the WASM fallback might only analyze 2 frames per second (2 FPS), which is still sufficient to gauge general engagement over a 45-minute interview.

## 3. Evaluated Alternatives

### 3.1 Transformers.js
An excellent library for running HuggingFace models directly in the browser.
- *Pros:* Extremely flexible, supports a massive variety of modern transformer architectures.
- *Cons:* FaceMesh and BlazeFace (the standard MediaPipe models) are highly specialized and optimized for this specific geometric task. General-purpose transformers often carry heavier weights and slower inference times for simple landmark detection compared to MediaPipe's purpose-built pipelines.

### 3.2 Server-Side Processing via Cloud API (e.g., AWS Rekognition)
Streaming frames from the browser directly to a third-party vision API.
- *Pros:* Solves the client-side hardware constraint. Works on any device.
- *Cons:* Extremely expensive. API providers charge per frame or per minute of video analyzed. Furthermore, this reintroduces the severe privacy liability of transmitting raw biometric data across the internet.

## 4. Telemetry and Heuristics

The neural network outputs raw geometric data (e.g., the exact XYZ coordinates of the left pupil, or a blendshape score for "jawOpen"). This raw data is useless for a post-interview report. 

The client-side architecture includes heuristic translation layers to convert geometry into actionable analytics before transmission.

### 4.1 Engagement Scoring
Engagement is calculated heuristically. Instead of training a heavy neural network to guess "is this person engaged?", the system uses simple, robust math based on the blendshapes:
- **Eye Contact:** Are the head yaw and pitch within +/- 15 degrees of the center? Are the eyelids open?
- **Active Listening:** Is the candidate demonstrating micro-expressions (nodding, slight brow movement) while the AI is speaking?
These metrics are combined into a normalized score (0.0 to 1.0) and transmitted to the server.

### 4.2 Cheating Detection (Multi-Face)
A critical feature of automated interviews is integrity. The vision model is configured to detect multiple faces. If a second set of facial landmarks enters the frame (suggesting someone else is in the room helping the candidate), the client immediately fires an alert payload to the server. The server can then instruct the LLM to dynamically address the situation (e.g., "I noticed someone else might be in the room, could you confirm you are alone?").
