from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings


def load_vectorstore():

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore



def retrieve_documents(question):

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=3
    )

    return results