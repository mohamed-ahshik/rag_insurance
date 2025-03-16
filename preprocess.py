import os
from typing import List, str

import pymupdf4llm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

data_folder = r"/Users/user/Documents/personal_projects/rag_insurance/data"


def get_pdf_files_in_folder(data_folder_dir: str) -> list:
    """
    Get all pdf files names in the folder

    Parameters:
    data_folder_dir (str): path to the folder

    Returns:
    pdf_files (list): list of pdf files in the folder

    """
    pdf_files = []
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            pdf_files.append(file)
    print("There are {} pdf files in the folder".format(len(pdf_files)))

    return pdf_files


def convert_pdf_to_markdown(pdf_files: list, processed_data_dir: str) -> None:
    """
    Convert pdf files to markdown

    Parameters:
    pdf_files (pdf): list of pdf files names
    processed_data_dir (str): path to the folder where the markdown files will be saved

    """
    for files in pdf_files:
        pdf_file = os.path.join(data_folder, files)
        split_file_name = files.split(".")
        split_file_name[-1] = "md"
        markdown_file_name = ".".join(split_file_name)
        pdf_document = pymupdf4llm.to_markdown(pdf_file)
        save_path = os.path.join(processed_data_dir, markdown_file_name)
        with open(save_path, "w") as file:
            file.write(pdf_document)


def load_markdown_files(processed_data_dir: str) -> List:
    """
    Load markdown files

    Parameters:
    processed_data_dir (str): path to the folder where the markdown files are saved

    Returns:
    docs (List): list of markdown files in langchain Document object

    """
    loader = DirectoryLoader(processed_data_dir, glob="**/*.md")
    docs = loader.load()
    print(f"There are {len(docs)} documents loaded")

    return docs


def get_chunks(docs: List, chunk_size: int = 1000, chunk_overlap: int = 300):
    """
    Split documents into chunks with tokwn text splitter

    Parameters:
    docs (List): list of markdown files in langchain Document object

    returns:
    chunks (List): list of chunks in langchain Document object

    """
    # Split documents into chunks
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs)

    return chunks


def store_chunks_into_vectorstore(chunks: List) -> VectorStoreRetriever:
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create vector store
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    # Create vectorstore retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 15}
    )

    return retriever


def get_compressed_docs(
    retriever: VectorStoreRetriever,
) -> ContextualCompressionRetriever:
    """
    Use reranker to compress the documents

    Parameters:
    retriever (VectorStoreRetriever): vectorstore retriever
    """

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever
