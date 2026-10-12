from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# =========================================================
# 1. Load Environment Variables
# =========================================================

load_dotenv()


# =========================================================
# 2. Create LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================================================
# 3. Load PDF
# =========================================================

pdf_path = "data/Build_RAG_From_Scratch_Using_LangChain_AmanAI_Lab.pdf"

loader = PyPDFLoader(
    pdf_path
)

documents = loader.load()

print("\nPDF loaded successfully.")
print("Number of pages:", len(documents))


# =========================================================
# 4. Split PDF into Chunks
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print("Number of chunks:", len(chunks))


# =========================================================
# 5. Create Embeddings
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 6. Create Vector Database
# =========================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector database created.")


# =========================================================
# 7. Create Retriever
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# =========================================================
# 8. Create Retrieval Tool
# =========================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the PDF knowledge base and return
    relevant information.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


# =========================================================
# 9. Bind Tool to LLM
# =========================================================

llm_with_tools = llm.bind_tools(
    [
        search_knowledge_base
    ]
)


# =========================================================
# 10. Agent
# =========================================================

def agent(user_query):

    messages = [

        (
            "system",

            """
You are an AI assistant.

You have access to a PDF knowledge base.

IMPORTANT:

1. If the user asks a question that requires
   information from the PDF, use the
   search_knowledge_base tool.

2. If the question is general and does not
   require information from the PDF, answer
   directly without using the tool.

3. Do not use the knowledge base unnecessarily.
"""
        ),

        (
            "user",
            user_query
        )
    ]


    # =====================================================
    # First LLM Call
    # =====================================================

    response = llm_with_tools.invoke(
        messages
    )


    # =====================================================
    # Check Agent Decision
    # =====================================================

    if not response.tool_calls:

        print(
            "\nAgent Decision:"
        )

        print(
            "No retrieval required."
        )

        print(
            "Answering directly using the LLM."
        )

        return response.content


    # =====================================================
    # Retrieval Required
    # =====================================================

    print(
        "\nAgent Decision:"
    )

    print(
        "Knowledge base retrieval required."
    )


    # Add agent response
    messages.append(
        response
    )


    # =====================================================
    # Execute Tool
    # =====================================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]


        print(
            "\nTool Selected:"
        )

        print(
            tool_name
        )


        print(
            "\nTool Arguments:"
        )

        print(
            tool_args
        )


        # ---------------------------------------------
        # Execute Retriever
        # ---------------------------------------------

        if tool_name == "search_knowledge_base":

            result = search_knowledge_base.invoke(
                tool_args
            )


            print(
                "\nRetrieved Information:"
            )

            print(
                result
            )


            # -----------------------------------------
            # Add Tool Result
            # -----------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                }
            )


    # =====================================================
    # Final LLM Call
    # =====================================================

    print(
        "\nGenerating final answer..."
    )


    final_response = llm_with_tools.invoke(
        messages
    )


    return final_response.content


# =========================================================
# 11. User Input
# =========================================================

question = input(
    "\nAsk a question: "
)


# =========================================================
# 12. Run Agent
# =========================================================

answer = agent(
    question
)


# =========================================================
# 13. Final Answer
# =========================================================

print(
    "\n======================================"
)

print(
    "FINAL ANSWER"
)

print(
    "======================================"
)

print(
    answer
)