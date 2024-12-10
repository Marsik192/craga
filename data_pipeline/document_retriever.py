from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


def create_document_retriever(
    vectorstore,
    num_documents=3
):
    """
    Create a retriever that will be used by the agent to retrieve documents.

    Args:
        vectorstore: vectorstore containing chunks of documents
        num_documents: number of documents to retrieve
    
    Returns:
        Ensemble of vectorstore backed retriever and BM25Retriever
    """

    vectorstore_retriever = vectorstore.as_retriever()

    all_docs = vectorstore.get()
    prepared_documents = []
    for i in range(len(all_docs["ids"])):
        document = Document(
            page_content=all_docs["documents"][i],
            metadata=all_docs["metadatas"][i]
        )
        prepared_documents.append(document)

    bm25_retriever = BM25Retriever.from_documents(prepared_documents)
    bm25_retriever.k = num_documents

    retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return retriever