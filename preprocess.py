from typing import List

import pymupdf4llm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def convert_pdf_to_markdown(pdf_file: str) -> str:
    """
    Convert pdf file to markdown

    Parameters:
    pdf_files (str): path to the pdf file
    processed_pdf_file (str): path to the folder where the markdown files will be saved

    """
    split_file_name = pdf_file.split(".")
    split_file_name[-1] = "md"
    markdown_file_path = ".".join(split_file_name)
    pdf_document = pymupdf4llm.to_markdown(pdf_file)
    with open(markdown_file_path, "w") as file:
        file.write(pdf_document)

    return markdown_file_path


def load_markdown_files(processed_data_path: str) -> List:
    """
    Load markdown files

    Parameters:
    processed_data_dir (str): path to the folder where the markdown files are saved

    Returns:
    docs (List): list of markdown files in langchain Document object

    """
    loader = UnstructuredMarkdownLoader(processed_data_path)
    docs = loader.load()

    return docs


def get_chunks(docs: List, chunk_size: int = 500, chunk_overlap: int = 200):
    """
    Split documents into chunks with tokwn text splitter

    Parameters:
    docs (List): list of markdown files in langchain Document object

    returns:
    chunks (List): list of chunks in langchain Document object

    """
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    return chunks


def store_chunks_into_vectorstore(chunks: List) -> VectorStoreRetriever:
    # Create embeddings
    model_kwargs = {"trust_remote_code": True}
    embeddings = HuggingFaceEmbeddings(
        model_name="Lajavaness/bilingual-embedding-large",
        show_progress=True,
        model_kwargs=model_kwargs,
    )
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

    # model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-electra-base")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever


def run_preprocess(pdf_path: str) -> ContextualCompressionRetriever:
    markdown_file_path = convert_pdf_to_markdown(pdf_path)
    loaded_markdown_doc = load_markdown_files(markdown_file_path)
    chunks = get_chunks(loaded_markdown_doc)
    retriever = store_chunks_into_vectorstore(chunks)
    compression_retriever = get_compressed_docs(retriever)

    return compression_retriever
