# Lab 1: Building a RAG Pipeline

## Objective
Build a Retrieval Augmented Generation (RAG) system.
We will index a PDF and query it.

## 1. Setup

```bash
poetry add chromadb langchain langchain-openai pypdf
```

## 2. The Pipeline (`rag.py`)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load & Split
loader = PyPDFLoader("handbook.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 2. Index (Vector DB)
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# 3. Retrieve & Generate
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 4. Query
query = "What is the vacation policy?"
result = qa_chain.invoke({"query": query})
print(result["result"])
```

## 3. Challenge
*   **Source Citations:** Modify the code to print the `page_number` of the retrieved chunks.
*   **Multi-Query:** Use an LLM to generate 3 variations of the user's query to improve retrieval recall.

## 4. Submission
Submit the code with Source Citations implemented.
