from langchain_community.vectorstores import FAISS


def similarity_retrieval(vectorstore, query):

    return vectorstore.similarity_search(
        query,
        k=3
    )



def mmr_retrieval(vectorstore, query):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":3
        }
    )

    return retriever.invoke(query)