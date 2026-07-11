# Security Checklist

This system handles sensitive user data (resumes, video metrics, voice transcripts). The following security measures are implemented.

## Application Security
- [x] **HTTPS Everywhere:** All traffic routes through Nginx with SSL (Let's Encrypt).
- [x] **JWT Auth:** Access tokens expire in 15 minutes.
- [x] **Password Hashing:** `bcrypt` used for all user passwords.
- [x] **CORS:** FastAPI CORS middleware strictly whitelists the frontend domain.
- [x] **Input Validation:** Pydantic models validate all incoming REST and WebSocket JSON payloads.
- [x] **SQL Injection:** SQLAlchemy ORM prevents injection attacks.
- [x] **Rate Limiting:** In-memory or Redis counter restricts `/api/auth/login` to prevent brute force.

## Server Security (Hostinger VPS)
- [x] **Firewall (UFW):** Only ports 22, 80, and 443 are open.
- [x] **SSH Access:** Root login disabled. Key-based authentication only.
- [x] **Secrets:** Passwords and API keys stored in `.env` (excluded from git).
- [x] **Ollama Cloud Key:** Kept securely in the FastAPI backend env vars, **never** exposed to the Next.js frontend.

## Data Privacy (GDPR/CCPA Compliance)
- [x] **Video Privacy:** **Raw video never leaves the client's browser.** Only computed numbers (e.g., `engagement: 0.8`) are sent to the server.
- [x] **Audio Privacy:** Audio chunks are held in RAM for STT processing and immediately discarded.
- [x] **Transcripts:** Saved in PostgreSQL for report generation. A `/api/users/me/delete` endpoint is provided to purge all user records.
- [x] **Resumes:** Stored on the local VPS file system, accessible only via authenticated API endpoints.
