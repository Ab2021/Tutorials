# Lab: Day 43 - Build a RAG Pipeline

## Goal
Chat with a text file.

## Prerequisites
- `pip install langchain langchain-openai chromadb`
- A text file `data.txt` (Create one with some dummy facts, e.g., "The secret code is 12345.").

## Step 1: The Code (`rag.py`)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load
loader = TextLoader("data.txt")
docs = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Store (Chroma)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 4. RAG Chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 5. Chat
print("--- RAG Bot ---")
print(chain.invoke("What is the secret code?"))
```

## Step 2: Run It
`python rag.py`

*   **Output**: "The secret code is 12345."

## Challenge
Replace `TextLoader` with `WebBaseLoader` (requires `beautifulsoup4`).
Point it to a URL (e.g., a Wikipedia page).
Ask questions about that page.
