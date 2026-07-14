from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


print("✅ Vector database loaded successfully!")


# Ask question
question = input("\nAsk a question: ")


# Retrieve documents
results = vectorstore.similarity_search(
    question,
    k=3
)


# Create context
context = ""

for doc in results:
    context += doc.page_content + "\n\n"


# Create prompt
prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""


# Send to Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)


# Print final answer
answer = response.choices[0].message.content


print("\n🤖 Answer:")
print(answer)