from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

def recursive_chunking(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)



def fixed_chunking(documents):

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)



from langchain_core.documents import Document


def sentence_chunking(documents):

    chunks = []

    for doc in documents:
        sentences = doc.page_content.split(".")

        for sentence in sentences:
            if sentence.strip():
                chunks.append(
                    Document(
                        page_content=sentence.strip(),
                        metadata=doc.metadata
                    )
                )

    return chunks