# Lab: Day 25 - API Docs

## Goal
Master the art of automated documentation. You will annotate a FastAPI app to produce a beautiful Swagger UI, then export the spec.

## Directory Structure
```
day25/
├── app.py
└── requirements.txt
```

## Step 1: The App (`app.py`)

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

app = FastAPI(
    title="My Awesome API",
    description="This is a sample API for Day 25 of the Backend Course.",
    version="1.0.0",
    docs_url="/docs", # Swagger UI
    redoc_url="/redoc" # ReDoc UI
)

class Item(BaseModel):
    name: str = Field(..., description="The name of the item", example="Widget")
    price: float = Field(..., gt=0, description="Price in USD", example=9.99)
    tags: list[str] = Field(default=[], example=["new", "sale"])

@app.post("/items", summary="Create an Item", response_description="The created item")
def create_item(
    item: Item = Body(..., embed=True)
):
    """
    Create a new item in the database.
    
    - **name**: must be unique
    - **price**: must be positive
    """
    return item
```

## Step 2: Run It

1.  **Run**: `uvicorn app:app --reload`
2.  **Swagger UI**: Visit `http://localhost:8000/docs`.
    *   Notice the Title, Description, and Example values in the schema.
3.  **ReDoc**: Visit `http://localhost:8000/redoc`.
    *   A different, cleaner view.

## Step 3: Export Spec

Get the raw JSON:
`curl http://localhost:8000/openapi.json > openapi.json`

## Challenge (Optional)
Use **OpenAPI Generator** (requires Java or Docker) to generate a client library.
```bash
docker run --rm -v "${PWD}:/local" openapitools/openapi-generator-cli generate \
    -i /local/openapi.json \
    -g python \
    -o /local/python-client
```
Now you have a full Python SDK for your API!
