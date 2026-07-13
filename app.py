from chunking import (
    recursive_chunking,
    fixed_chunking,
    sentence_chunking
)

from retrievers import (
    similarity_retrieval,
    mmr_retrieval
)

from embeddings import get_embeddings
from llm import generate_answer

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# -----------------------------
# Load documents (PDFs)
# -----------------------------

loader = PyPDFDirectoryLoader(
    "data"
)

documents = loader.load()

print("Number of documents loaded:", len(documents))


if len(documents) == 0:
    print("❌ No documents found. Check your data folder.")
    exit()


# -----------------------------
# Load embedding model once
# -----------------------------

embeddings = get_embeddings()


# -----------------------------
# Apply 3 chunking methods
# -----------------------------

chunking_methods = {

    "Recursive Chunking": recursive_chunking(documents),

    "Fixed Chunking": fixed_chunking(documents),

    "Sentence Chunking": sentence_chunking(documents)

}


# -----------------------------
# Ask question
# -----------------------------

question = input("\nAsk a question: ")


# Store outputs
outputs = {}


# -----------------------------
# Run 3 chunking × 2 retrieval
# -----------------------------

for chunk_name, chunks in chunking_methods.items():

    print(f"\nProcessing {chunk_name}...")

    if len(chunks) == 0:
        print(f"❌ No chunks created for {chunk_name}")
        continue


    # Create vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # =============================
    # Retrieval 1: Similarity Search
    # =============================

    similarity_docs = similarity_retrieval(
        vectorstore,
        question
    )


    similarity_context = ""

    for doc in similarity_docs:
        similarity_context += doc.page_content + "\n\n"


    similarity_answer = generate_answer(
        question,
        similarity_context
    )


    outputs[
        f"{chunk_name} + Similarity Retrieval"
    ] = similarity_answer



    # =============================
    # Retrieval 2: MMR Search
    # =============================

    mmr_docs = mmr_retrieval(
        vectorstore,
        question
    )


    mmr_context = ""

    for doc in mmr_docs:
        mmr_context += doc.page_content + "\n\n"


    mmr_answer = generate_answer(
        question,
        mmr_context
    )


    outputs[
        f"{chunk_name} + MMR Retrieval"
    ] = mmr_answer



# -----------------------------
# Display all results
# -----------------------------

print("\n\n========== RESULTS ==========\n")


for method, answer in outputs.items():

    print("=" * 70)
    print(method)
    print("=" * 70)

    print(answer)

    print("\n")