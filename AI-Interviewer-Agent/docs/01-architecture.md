# Comprehensive System Architecture Design and Analysis (2026 Edition)

This document provides a deep, exhaustive analysis of the system architecture for the AI Interviewer Agent. The design is specifically constrained by the target deployment environment: a Hostinger Virtual Private Server (VPS) equipped with 2 vCPUs, 8GB of RAM, and a 100GB SSD, completely lacking dedicated GPU hardware. 

## 1. The Hybrid Architecture Paradigm

The core philosophy of this architecture is the "Hybrid Edge-Cloud" model. In this context, the candidate's browser acts as the "Ultra-Edge," the Hostinger VPS acts as the "Edge," and Ollama Cloud acts as the "Heavy Cloud." 

### 1.1 Architectural Split
- **The Ultra-Edge (Browser):** Handles media capture, Voice Activity Detection (VAD), and Video Analysis (facial landmarks, engagement metrics).
- **The Edge (Hostinger VPS):** Acts as the orchestrator, managing connections, database persistence, Speech-to-Text (STT) transcription, and Text-to-Speech (TTS) synthesis.
- **The Heavy Cloud (Ollama Cloud):** Exclusively handles Large Language Model (LLM) inference.

### 1.2 Pros of the Hybrid Approach
- **Cost Efficiency:** Utilizes a $9/month Hostinger VPS for 90% of the workload.
- **Latency Optimization:** STT and TTS are the most latency-sensitive components. Running them locally minimizes network hops. 
- **Privacy Preservation:** Raw video NEVER leaves the client.

### 1.3 Cons and Risks of the Hybrid Approach
- **Complex Orchestration:** A distributed pipeline (Browser -> VPS -> Cloud -> VPS -> Browser).
- **CPU Bottlenecks:** 2 vCPUs are easily overwhelmed by concurrent STT/TTS requests.

---

## 2. Component Analysis: Orchestration and Networking (2026 Update)

Real-time audio requires a robust networking protocol. As of 2026, the landscape offers three primary choices: WebSockets, WebRTC, and WebTransport.

### 2.1 The Selection: WebSockets (Transitioning to WebTransport)
Currently, the MVP utilizes WebSockets (WSS) proxied through Nginx.

- **Pros:** 
  - Ubiquitous browser support and native integration in FastAPI.
  - Extremely simple to implement compared to WebRTC signaling.
- **Cons (The Head-of-Line Blocking Problem):** 
  - WebSockets operate over TCP. In real-time audio, if a single packet drops, the entire TCP stream stalls while waiting for retransmission. This causes audio stuttering.

### 2.2 Future-Proofing: WebTransport (HTTP/3)
In 2026, **WebTransport** has achieved Baseline status across all major browsers. The architecture is designed to swap the WebSocket handler for WebTransport.
- **Why WebTransport?** It utilizes QUIC/HTTP/3 (UDP-based). It solves the Head-of-Line blocking problem of WebSockets, providing the raw speed of UDP without the massive infrastructural overhead of WebRTC (which requires STUN/TURN servers and complex SDP negotiation). 
- **WebTransport over WebRTC:** While WebRTC is the gold standard for voice, deploying an SFU (Selective Forwarding Unit) on a 2 vCPU VPS is overkill. WebTransport provides a perfect middle ground for client-to-server AI streaming.

---

## 3. Component Analysis: Memory and CPU Budgeting

Running multiple AI models on an 8GB RAM machine requires strict resource management.

### 3.1 Strict Docker Resource Limits
- **FastAPI / AI Container (2.5GB Limit):** Python runtime + Sherpa-ONNX + Kokoro.
- **PostgreSQL (512MB Limit):** Tuned `shared_buffers`.
- **Next.js Frontend (256MB Limit):** Lightweight Node.js rendering.
- **Redis (128MB Limit):** Ephemeral WebSocket session state.

---

## 4. Component Analysis: State Management

### 4.1 Ephemeral State (Redis)
During an active interview, current state ("candidate speaking", "LLM generating") lives in Redis, ensuring the FastAPI event loop isn't blocked by disk I/O.

### 4.2 Persistent State (PostgreSQL)
All transcripts and video metrics are flushed to PostgreSQL for ACID compliance and long-term analytics.

---

## 5. Security and Network Topology

Nginx acts as the reverse proxy for SSL termination. All LLM requests to Ollama Cloud are securely proxied through FastAPI, ensuring API keys are never exposed to the frontend browser.
