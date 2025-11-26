# Lab: Day 10 - Secure Auth with JWT

## Goal
Build a secure Authentication system. You will implement User Registration (hashing passwords) and Login (issuing JWTs), then protect an endpoint.

## Prerequisites
- Python + `fastapi` + `uvicorn` + `passlib[bcrypt]` + `python-jose`.

## Directory Structure
```
day10/
├── main.py
└── requirements.txt
```

## Step 1: Requirements

```text
fastapi
uvicorn
passlib[bcrypt]
python-jose[cryptography]
```

## Step 2: The Auth Server (`main.py`)

```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

# Configuration
SECRET_KEY = "super-secret-key-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-Memory DB
fake_users_db = {}

# Models
class User(BaseModel):
    username: str
    password: str # Plain text (only for request)

class Token(BaseModel):
    access_token: str
    token_type: str

# Helpers
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Endpoints

@app.post("/register")
def register(user: User):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    hashed_pw = get_password_hash(user.password)
    fake_users_db[user.username] = {
        "username": user.username,
        "hashed_password": hashed_pw
    }
    return {"msg": "User created successfully"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected Route
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user, "msg": "You are authorized!"}
```

## Step 3: Test It

1.  **Run**: `uvicorn main:app --reload`
2.  **Open Swagger UI**: `http://localhost:8000/docs`
3.  **Register**: `/register` with `{"username": "alice", "password": "secret"}`.
4.  **Login**: Click "Authorize" button (top right) or use `/token` endpoint.
5.  **Access Protected**: Try `/users/me`.

## Challenge
Add a `role` field to the user and token. Create an endpoint `/admin` that only allows users with `role="admin"`.
