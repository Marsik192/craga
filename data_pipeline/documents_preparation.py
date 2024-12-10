import os

import nltk
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def prepare_table_elements(documents):
    """Merge table element with the previous element in order to provide context to tables.

    Args:
        documents: A list of Document objects.
    
    Returns:
        A list of new Document objects.
    """

    cleaned_documents = []

    for document in documents:
        if "category" in document.metadata:
            category = document.metadata["category"]
            if category == "Table" and len(cleaned_documents) > 0:
                new_doc = cleaned_documents[-1] # Pick the last document
                new_doc.page_content += document.page_content
                cleaned_documents = cleaned_documents[:-1] # Remove the last document
                cleaned_documents.append(new_doc)
            else:
                cleaned_documents.append(document)
        else:
            cleaned_documents.append(document)

    return cleaned_documents

def change_source_metadata(documents, filename):
    """
    Add given filename to the metadata of the documents if it is not already added.
    
    Args:
        documents: A list of Document objects.
        filename: A name of the file.
    
    Returns:
        A list of updated Document objects.
    """
    for document in documents:

        # Add filename metadata
        if "filename" not in document.metadata:
            document.metadata.update({"filename": filename})
        
        # Change source metadata to show only filename
        document.metadata.update({"source": filename})
    return documents 

def load_single_document(file):
    """
    Load a document using the UnstructuredFileLoader and extract images if the document is a PDF.

    Args:
        file: The path to the file to be loaded or bytes.
        images_dir: The directory to save the extracted images.

    Returns:
        A list of Document objects. 
    """

    loader = UnstructuredLoader(
            file=file,
            strategy="fast",
            mode="elements",
            chunking_strategy="by_title",
            max_characters=1700,
            new_after_n_chars=1500,
            combine_text_under_n_chars=1000,
            skip_infer_table_types=[],
            metadata_filename="data.pdf"
        )

    documents = loader.load()
    
    # Concat table elements to the previos context
    cleaned_documents = prepare_table_elements(documents)

    # Add filename to the metadata (required if "chunk_mode" == "single")
    cleaned_documents = change_source_metadata(cleaned_documents, "data.pdf")

    cleaned_documents_metadata = filter_complex_metadata(cleaned_documents)

    return cleaned_documents_metadata