# Lab: Day 52 - Core API

## Goal
Build the REST API.

## Prerequisites
- `pip install fastapi uvicorn sqlalchemy psycopg2-binary pyjwt passlib kafka-python`

## Step 1: Database (`database.py`)

```python
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/documind"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    content = Column(Text)
```

## Step 2: Kafka Producer (`events.py`)

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def publish_doc_update(doc_id, content):
    producer.send('doc_updates', {'doc_id': doc_id, 'content': content})
```

## Step 3: API (`main.py`)

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal, Document
from events import publish_doc_update
from pydantic import BaseModel

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DocCreate(BaseModel):
    title: str
    content: str

@app.post("/docs/")
def create_doc(doc: DocCreate, db: Session = Depends(get_db)):
    db_doc = Document(title=doc.title, content=doc.content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    # Publish Event
    publish_doc_update(db_doc.id, db_doc.content)
    
    return db_doc
```

## Step 4: Run It
1.  `uvicorn main:app --reload`
2.  Create a doc via Swagger UI (`http://localhost:8000/docs`).
3.  Check Kafka Consumer (from previous labs) to see the event.

## Challenge
Implement **Authentication**.
1.  Add `User` table.
2.  Add `/login` endpoint.
3.  Protect `/docs/` with `Depends(get_current_user)`.
