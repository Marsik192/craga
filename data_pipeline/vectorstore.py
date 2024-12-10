import os

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma


class VectorStore:
    """
    A class for managing a ChromaDB vectorstore.

    Args:
        vectorstore_dir: The working directory of vectorstore data.
    """

    def __init__(self, vectorstore_dir):
        self.vectorstore_dir = vectorstore_dir
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self.load_vectorstore()

    def create_vectorstore(self, documents):
        """
        Create a vectorstore to embed and store Document objects.

        Args:
            documents: A list of Document objects.
        """

        # Split documents to avoid Chroma batch limit
        batched_documents = self.split_list(documents)
        
        for batch in batched_documents:
            self.vectorstore = Chroma.from_documents(
                batch,
                embedding=self.embedding_function,
                persist_directory=self.vectorstore_dir
            )
            self.vectorstore.persist()

        print("Vectorstore created succesfully")
        return self.vectorstore
    
    def clear_vectorstore(self):
        """
        Clear the vectorstore
        """

        if self.vectorstore is not None:
            all_ids = self.vectorstore.get()["ids"]
            if len(all_ids) != 0:
                batched_ids = self.split_list(all_ids)
                for batch in batched_ids:
                    self.vectorstore.delete(batch)

    def load_vectorstore(self):
        """
        Load the vectorstore from a folder
        """

        # Load vectorstore if it exists
        if os.path.isdir(self.vectorstore_dir):
            vectorstore = Chroma(
                persist_directory=self.vectorstore_dir, 
                embedding_function=self.embedding_function
            )
            print("Vectorstore loaded")

            return vectorstore
        
        return None
    
    def split_list(self, input_list, chunk_size=5000):
        """
        Returns a generator with batches of the list in chunk_size length.

        Args:
            input_list: A list to be split.
            chunk_size: A size of the chunk.
        
        Returns:
            A generator with batches of the list in chunk_size length.
        """
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]