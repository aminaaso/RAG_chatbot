A Retrieval-Augmented Generation (RAG) chatbot built using the AI Season Bootcamp documents as its knowledge base.
This project explores how a Large Language Model (LLM) can answer questions more accurately when it is provided with relevant information retrieved from external documents.
Instead of relying only on the LLM's existing knowledge, this chatbot first searches through the AI Season documents, finds the most relevant information, and then uses that context to generate a grounded response.

About the Project:
Large Language Models are powerful, but they have limitations:
They may not know private or newly updated information. They can sometimes generate incorrect answers. They do not automatically have access to custom documents.

To solve this problem, I built a RAG pipeline where:
1. Documents are loaded and processed.
2. Text is divided into smaller chunks.
3. Chunks are converted into embeddings.
4. The embeddings are stored in a FAISS vector database.
5. Relevant chunks are retrieved based on the user's question.
6. A Groq-powered LLM generates the final answer using the retrieved context.

What This Project Does:
This chatbot compares different RAG approaches by testing:

Chunking Strategies Used:
1. Recursive Chunking
Splits documents while trying to maintain the meaning and structure of the text.

2. Fixed Chunking
Splits documents into fixed-size text segments.

3. Sentence Chunking
Splits documents sentence by sentence to preserve smaller pieces of information.

Retrieval Techniques Used:

1. Similarity Search
Finds document chunks that are most similar to the user's query.

2. Maximum Marginal Relevance (MMR)
Retrieves relevant information while reducing repeated content.

Technologies Used:
1. Python
2. LangChain
3. FAISS Vector Database
4. HuggingFace Sentence Transformers
5. Groq LLM API
6. PyPDF Loader
7. dotenv

Key Learnings:
1. How RAG systems work internally
2. Why document chunking affects retrieval quality
3. How embeddings represent text as vectors
4. How vector databases enable semantic search
5. The difference between similarity search and MMR retrieval
6. How LLMs can generate better answers when provided with relevant context

Some possible future improvements:
1. Add a Streamlit web interface
2. Add conversation memory
3. Add document upload functionality
4. Add answer citations with source pages
5. Deploy the chatbot online


