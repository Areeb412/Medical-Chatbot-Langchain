from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def load_pdf(data_path):
    """Load all PDFs from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    """Split documents into overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_embeddings():
    """Return an OpenAI embeddings client."""
    embeddings = OpenAIEmbeddings()
    return embeddings