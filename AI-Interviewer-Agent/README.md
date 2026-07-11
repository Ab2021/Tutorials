# AI Interviewer Agent

A real-time, multimodal conversational AI interviewer with voice and video analysis capabilities.

## Tech Stack Summary

| Component | Technology | Notes |
| :--- | :--- | :--- |
| **Frontend** | Next.js 15 | React framework, minimal UI |
| **Backend** | FastAPI (Python) | WebSocket server, async processing |
| **STT** | Vosk | Lightweight, CPU-friendly, native streaming |
| **TTS** | Piper | ONNX runtime, runs on CPU (~200MB RAM) |
| **LLM** | Ollama Cloud | Gemma 4 12B (conversation), Phi-4-mini (routing) |
| **Video** | MediaPipe FaceMesh | Client-side only (browser) |
| **Database**| PostgreSQL + Redis | Persistence and session state |
| **Hosting** | Hostinger VPS | 2 vCPU, 8GB RAM, 100GB SSD (~$9/mo) |

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd AI-Interviewer-Agent

# Copy env template and fill in values (especially Ollama Cloud API key)
cp .env.example .env

# Start all services
docker compose up -d
```

## Project Structure

```
AI-Interviewer-Agent/
├── docs/                  # Detailed documentation
├── frontend/              # Next.js application
├── backend/               # FastAPI application
├── docker-compose.yml     # Multi-container deployment
└── README.md
```

## Documentation

See the `docs/` folder for detailed, component-specific documentation:
- [01 Architecture](./docs/01-architecture.md)
- [02 Speech-to-Text (STT)](./docs/02-stt.md)
- [03 Text-to-Speech (TTS)](./docs/03-tts.md)
- [04 Large Language Model (LLM)](./docs/04-llm.md)
- [05 Video Analysis](./docs/05-video-analysis.md)
- [06 Frontend](./docs/06-frontend.md)
- [07 Backend](./docs/07-backend.md)
- [08 Database](./docs/08-database.md)
- [09 API Design](./docs/09-api.md)
- [10 Deployment](./docs/10-deployment.md)
- [11 Security](./docs/11-security.md)
- [12 Latency Optimization](./docs/12-latency.md)
