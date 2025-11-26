# Lab: Day 31 - Google Login with FastAPI

## Goal
Implement "Login with Google" using OIDC.

## Prerequisites
- `pip install fastapi uvicorn authlib httpx starlette`
- A Google Cloud Project with OAuth Credentials (Client ID & Secret).
    - *Redirect URI*: `http://localhost:8000/auth`

## Step 1: The App (`app.py`)

```python
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

# Load env vars
config = Config('.env')
oauth = OAuth(config)

# Register Google
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="some-random-string")

@app.get("/")
def home(request: Request):
    user = request.session.get('user')
    if user:
        return {"message": f"Hello, {user['name']}!", "email": user['email']}
    return {"message": "Please login", "link": "/login"}

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user = token.get('userinfo')
        if user:
            request.session['user'] = user
        return {"status": "success", "user": user}
    except Exception as e:
        return {"error": str(e)}

@app.get("/logout")
def logout(request: Request):
    request.session.pop('user', None)
    return {"message": "Logged out"}
```

## Step 2: The Config (`.env`)
Create a `.env` file:
```ini
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
```

## Step 3: Run It
1.  `uvicorn app:app --reload`
2.  Go to `http://localhost:8000`.
3.  Click Login -> Redirects to Google -> Back to App.
4.  See your name and email.

## Challenge
Implement **MFA (TOTP)**.
1.  Use `pyotp` library.
2.  Generate a secret for the user: `pyotp.random_base32()`.
3.  Show QR code (using `qrcode` lib).
4.  Verify code: `totp.verify(user_input)`.
