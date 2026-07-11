# Backend Architecture

The backend is built with **FastAPI** (Python 3.12). It acts as the central orchestrator, handling WebSockets, STT/TTS routing, LLM communication, and database operations.

## Project Structure

A flat, simple structure is used to minimize complexity:

```text
backend/
├── main.py
├── config.py
├── requirements.txt
├── Dockerfile
├── routers/
│   ├── auth.py
│   ├── interviews.py
│   └── health.py
├── ws/
│   ├── handler.py        # Manages WebSocket lifecycle
│   └── protocol.py       # Defines JSON message types
├── services/
│   ├── stt_service.py    # Wrapper for Vosk
│   ├── tts_service.py    # Wrapper for Piper
│   ├── llm_service.py    # Wrapper for Ollama Cloud
│   ├── interview.py      # The core orchestrator pipeline
│   └── scoring.py        # Simple rubric-based evaluator
├── models/
│   ├── schemas.py        # Pydantic models (Validation)
│   └── db.py             # SQLAlchemy models (Database)
└── db/
    └── database.py       # PostgreSQL connection setup
```

## Key Dependencies

Keep `requirements.txt` minimal:

```text
fastapi
uvicorn[standard]
websockets
aiohttp
python-multipart
sqlalchemy[asyncio]
asyncpg
alembic
redis
python-jose[cryptography]
passlib[bcrypt]
vosk
piper-tts
pydantic
pydantic-settings
```

*Note: We strictly avoid heavyweight frameworks like Celery. We use FastAPI's built-in `BackgroundTasks` for asynchronous report generation.*

## Interview Orchestrator Pipeline

The core logic lives in `services/interview.py`. It is a continuous async loop:

1. Receive WebSocket audio chunk.
2. Pass chunk to `stt_service` (Vosk).
3. If Vosk returns `is_final: True` (candidate stopped speaking):
   - Format transcript and add to history.
   - Send prompt to `llm_service`.
4. As `llm_service` streams text chunks back:
   - Buffer text into complete sentences.
   - Send complete sentence to `tts_service` (Piper).
5. Send Piper's PCM audio back via WebSocket to the frontend.

## WebSocket Protocol

All JSON messages over the WebSocket contain a `type` field.

- `audio_chunk`: PCM audio from candidate to server.
- `video_analysis`: JSON metadata from candidate to server.
- `transcript`: Partial or final STT text from server to candidate.
- `audio_response`: PCM audio from Piper from server to candidate.
- `event`: e.g., `question_asked`, `interview_end`.
