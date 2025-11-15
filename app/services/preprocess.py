import os
import sys
from pathlib import Path
from typing import List, Any

from dotenv import load_dotenv
from langchain.schema import BaseRetriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params

# Load environment variables
load_dotenv()

os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# Load config parameters
params = load_params("config/params.yaml")
preprocess_params = params.get("preprocess_params", {})

log_file_path = preprocess_params.get("log_file_path", "docs_preprocess.log")
chunk_size = preprocess_params.get("chunk_size", 1000)
chunk_overlap = preprocess_params.get("chunk_overlap", 150)

logger = setup_logger("DocumentPreprocessor", log_file_path)


class DocumentPreprocessor:
    """
    Handles document ingestion, preprocessing, and vectorization
    for RAG (Retrieval-Augmented Generation) pipelines.

    Responsibilities:
        - Load documents (PDFs, text files)
        - Split documents into semantic chunks
        - Generate vector embeddings
        - Store vectors in Pinecone for retrieval

    Typical usage example:
        >>> preprocessor = DocumentPreprocessor()
        >>> docs = preprocessor.load_documents(uploaded_files)
        >>> retriever = preprocessor.build_retriever(docs)
    """

    def __init__(self):
        """Initialize embedding and Pinecone configuration."""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_VECTORS_INDEX_NAME", "agent-vector-store")

        if not self.pinecone_api_key:
            raise AppException("PINECONE_API_KEY not found in environment.", sys)
        
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key

        logger.info("DocumentPreprocessor initialized successfully.")


    def load_documents(self, sources: List[str]) -> List[Any]:
        """
        Load and parse documents from a list of sources (file paths or URLs).

        Args:
            sources (List[str]): List of file paths or URLs.

        Returns:
            List[Any]: A list of LangChain Document objects.
        """
        all_documents = []

        if not sources:
            logger.warning("No sources were provided for processing.")
            return []

        for source in sources:
            source_name = os.path.basename(source) if not source.startswith('http') else source
            try:
                # Handle URLs and file paths
                if source.startswith("http://") or source.startswith("https://"):
                    logger.info(f"Loading content from URL: {source_name}")
                    loader = WebBaseLoader(source)
                    documents = loader.load()
                
                # Handle file paths
                else:
                    file_path = Path(source)
                    file_extension = file_path.suffix.lower()
                    
                    if file_extension == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        documents = loader.load()
                    elif file_extension in [".txt", ".text"]:
                        loader = TextLoader(str(file_path))
                        documents = loader.load()
                    else:
                        logger.warning(f"Unsupported file extension: {file_extension}. Skipping '{source_name}'.")
                        continue
                
                all_documents.extend(documents)
                logger.info(f"Successfully loaded document from '{source_name}'.")

            except Exception as e:
                logger.error(f"Error loading source '{source_name}': {e}", exc_info=True)
            

        logger.info(f"Successfully loaded content from {len(all_documents)} total documents.")
        return all_documents


    def build_retriever(self, loaded_docs: List[Any], session_id: str) -> BaseRetriever:
        """
        Process documents into chunks, create embeddings, and store them in Pinecone.

        Args:
            loaded_docs (List[Any]): List of LangChain Document objects.
            session_id (str): A unique identifier  session_id to isolate vectors.

        Returns:
            BaseRetriever: LangChain Retriever instance for querying similar chunks.
        """
        try:
            if not loaded_docs:
                logger.warning("No documents to process. Skipping retriever creation.")
                return None

            logger.info(f"Splitting documents into semantic chunks for the session_id: {session_id}")
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(loaded_docs)
            logger.info(f"Document split into {len(chunks)} chunks.")

            logger.info("Initializing MiniLM embedding model...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            logger.info(f"Storing document vectors into Pinecone under session_id: {session_id}")

            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                index_name=self.index_name,
                namespace=session_id
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 5,  "namespace": session_id})
            logger.info(f"Retriever created successfully with Pinecone vector store for the session_id: {session_id}.")

            return retriever

        except Exception as e:
            logger.error(f"Error during document preprocessing or Pinecone vectorization: {e}")
            raise AppException(e, sys)
