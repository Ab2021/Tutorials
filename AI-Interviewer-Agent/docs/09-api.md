# API Documentation

The system uses a minimal set of REST endpoints for state management and a single WebSocket endpoint for the real-time interview loop.

## Authentication

All protected endpoints require a JWT token in the `Authorization: Bearer <token>` header.
- **Access Token:** 15-minute expiry.
- **Refresh Token:** 7-day expiry (stored in `httpOnly` cookie).

## REST Endpoints (Minimal Set)

### Auth
- `POST /api/auth/register` — Create a new user.
- `POST /api/auth/login` — Authenticate and return JWT.

### Interviews
- `GET /api/interviews` — List current user's interviews.
- `POST /api/interviews` — Create a new interview session.
- `GET /api/interviews/{id}` — Get details of a specific interview.
- `GET /api/reports/{id}` — Fetch the post-interview evaluation report.

### File Upload
- `POST /api/upload/resume` — Upload candidate resume (saved to local disk, not S3).

### System
- `GET /api/health` — Basic health check (`{"status": "ok"}`).

## WebSocket Endpoint

The core real-time loop occurs over a single WebSocket connection.

```
WSS /ws/interview?token={jwt}&session_id={id}
```

### WebSocket Message Protocol

All messages are JSON objects containing a `type` field.

#### Client -> Server (Browser to FastAPI)

**1. Init**
```json
{
  "type": "init"
}
```

**2. Audio Chunk (PCM)**
```json
{
  "type": "audio_chunk",
  "data": "<base64_encoded_pcm_bytes>"
}
```

**3. Video Analysis Metadata**
```json
{
  "type": "video_analysis",
  "timestamp_s": 17000000.0,
  "engagement": 0.85,
  "emotions": {"happy": 0.2, "neutral": 0.8}
}
```

**4. End**
```json
{
  "type": "end"
}
```

#### Server -> Client (FastAPI to Browser)

**1. Transcript (STT Output)**
```json
{
  "type": "transcript",
  "speaker": "candidate",
  "text": "I have three years of experience...",
  "is_final": true
}
```

**2. Audio Response (TTS Output)**
```json
{
  "type": "audio_response",
  "data": "<base64_encoded_pcm_bytes>"
}
```

**3. Event**
```json
{
  "type": "event",
  "event_name": "question_asked",
  "payload": {"question_number": 2}
}
```
