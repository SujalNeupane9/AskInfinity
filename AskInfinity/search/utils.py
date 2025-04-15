# search/utils.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = '.................'   ## Add you api key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def create_vector_db_from_pdf(pdf_path, persist_directory=None):
    """
    Create a vector database from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Optional directory to save the vector database
        
    Returns:
        FAISS vector store object
    """
    # Load documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create and optionally save vector store
    if persist_directory:
        vector_store = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
        )
        vector_store.save_local(persist_directory)
    else:
        vector_store = FAISS.from_documents(splits, embeddings)
        
    return vector_store
