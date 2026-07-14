from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Read the extracted text
text_file = Path("docs") / "all_notes.txt"

with open(text_file, "r", encoding="utf-8") as file:
    text = file.read()

print("Total Characters:", len(text))

# Split the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Create embeddings
print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
print("Creating vector database...")

vectorstore = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

# Save the database
vectorstore.save_local("vectorstore")

print("\n✅ Vector database saved successfully!")