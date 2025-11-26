# Lab: Day 4 - Designing a RESTful Library API

## Goal
Design and implement a slightly more complex API with relationships. You will build a **Library API** managing **Authors** and **Books**.

## Requirements
1.  **Authors**: `id`, `name`, `bio`.
2.  **Books**: `id`, `title`, `year`, `author_id`.
3.  **Relationships**:
    *   Get all books by a specific author.
    *   Create a book for an author.

## Directory Structure
```
day04/
├── main.py
└── README.md
```

## The Code (`main.py`)

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# --- Models ---
class Author(BaseModel):
    id: Optional[int] = None
    name: str
    bio: Optional[str] = None

class Book(BaseModel):
    id: Optional[int] = None
    title: str
    year: int
    author_id: int

# --- In-Memory DB ---
authors_db = []
books_db = []
author_id_counter = 0
book_id_counter = 0

# --- Endpoints ---

# 1. Create Author
@app.post("/authors", response_model=Author, status_code=201)
def create_author(author: Author):
    global author_id_counter
    author_id_counter += 1
    author.id = author_id_counter
    authors_db.append(author)
    return author

# 2. List Authors
@app.get("/authors", response_model=List[Author])
def list_authors():
    return authors_db

# 3. Create Book (Nested or Flat? Let's go Flat for creation, but validate Author)
@app.post("/books", response_model=Book, status_code=201)
def create_book(book: Book):
    # Validate author exists
    if not any(a.id == book.author_id for a in authors_db):
        raise HTTPException(status_code=400, detail="Author ID does not exist")
    
    global book_id_counter
    book_id_counter += 1
    book.id = book_id_counter
    books_db.append(book)
    return book

# 4. Get Books (With Filtering)
# GET /books?year=2023
@app.get("/books", response_model=List[Book])
def list_books(year: Optional[int] = None):
    if year:
        return [b for b in books_db if b.year == year]
    return books_db

# 5. Get Books by Author (Sub-resource)
# GET /authors/{id}/books
@app.get("/authors/{author_id}/books", response_model=List[Book])
def get_author_books(author_id: int):
    # Check author exists
    if not any(a.id == author_id for a in authors_db):
        raise HTTPException(status_code=404, detail="Author not found")
    
    return [b for b in books_db if b.author_id == author_id]

# 6. Delete Author (And Cascade Delete Books?)
@app.delete("/authors/{author_id}", status_code=204)
def delete_author(author_id: int):
    global authors_db, books_db
    # Filter out the author
    initial_len = len(authors_db)
    authors_db = [a for a in authors_db if a.id != author_id]
    
    if len(authors_db) == initial_len:
        raise HTTPException(status_code=404, detail="Author not found")
    
    # Cascade delete books (Business Logic Decision)
    books_db = [b for b in books_db if b.author_id != author_id]
    return
```

## Lab Tasks

1.  **Run the Server**: `uvicorn main:app --reload`
2.  **Populate Data**:
    *   Create 2 Authors (e.g., "J.K. Rowling", "George R.R. Martin").
    *   Create Books for them.
    *   Try to create a book for a non-existent author (Expect 400).
3.  **Test Filtering**:
    *   `GET /books?year=1997`
4.  **Test Sub-resource**:
    *   `GET /authors/1/books`

## Challenge
Implement **Pagination** for the `/books` endpoint.
*   Add `skip` and `limit` query parameters.
*   Default `limit` to 10.
*   Slice the list: `books_db[skip : skip + limit]`.
