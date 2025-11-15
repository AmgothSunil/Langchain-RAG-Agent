import os
import sys
import hashlib
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params

load_dotenv()

# Load parameters from YAML
params = load_params("config/params.yaml")
pinecone_params = params.get("pinecone_memory_params", {})
log_file_path = pinecone_params.get("log_file_path", "pinecone_memory.log")

logger = setup_logger("PineconeMemory", log_file_path)


class PineconeMemory:
    """
    Production-grade semantic memory storage using Pinecone.
    
    This module handles:
        - Embedding user memory using HuggingFace models
        - Storing embeddings into Pinecone with metadata
        - Querying Pinecone for relevant long-term memories
    
    Components included:
        • Memory embedding
        • Memory upsert (store/update)
        • Memory retrieval (semantic search)
    
    Typical Usage:
        >>> memory = PineconeMemory()
        >>> memory.store_memory("sunil", "User loves TensorFlow.")
        >>> results = memory.retrieve_memory("sunil", "What ML framework?")
    """

    def __init__(self):
        """Initialize Pinecone client, index, and embedding model."""
        try:
            # Load API keys 
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.index_name = os.getenv("PINECONE_MEMORY_INDEX_NAME")

            if not self.pinecone_api_key:
                raise AppException("PINECONE_API_KEY is missing.", sys)

            if not self.index_name:
                raise AppException("PINECONE_MEMORY_INDEX_NAME is missing.", sys)

            logger.info("Initializing Pinecone client for memory store...")

            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.index_name)

            logger.info(f"Pinecone memory index loaded: {self.index_name}")

            # Initialize Embedding Model
            self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        except Exception as e:
            logger.error(f"Failed to initialize PineconeMemory: {e}")
            raise AppException(e, sys)


    # Embedding
    def embed_text(self, text: str) -> list:
        """
        Convert text into a vector embedding using HuggingFace.

        Args:
            text (str): The text to embed.

        Returns:
            list: Dense vector representation.
        """
        try:
            embedding = self.embedder.embed_query(text)
            return embedding  
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise AppException(e, sys)


    # Memory Storage
    def store_memory(self, user_id: str, memory_text: str):
        """
        Store a semantic memory vector into Pinecone with metadata.

        Args:
            user_id (str): Unique user identifier.
            memory_text (str): Clean extracted memory string.
        """
        try:
            vector = self.embed_text(memory_text)
            mem_id = f"{user_id}-{hashlib.md5(memory_text.encode()).hexdigest()}"

            self.index.upsert([
                {
                    "id": mem_id,
                    "values": vector,
                    "metadata": {
                        "user_id": user_id,
                        "memory": memory_text
                    }
                }
            ])

            logger.debug(f"[Memory Stored] user={user_id} | memory='{memory_text}'")

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise AppException(e, sys)


    # Memory Retrieval
    def retrieve_memory(self, user_id: str, query: str, top_k: int = 5) -> list:
        """
        Retrieve relevant long-term memories via semantic similarity search.

        Args:
            user_id (str): The user requesting memory.
            query (str): The current user input.
            top_k (int): Maximum number of memories to return.

        Returns:
            list[str]: List of memory strings.
        """
        try:
            vector = self.embed_text(query)

            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter={"user_id": user_id}
            )

            matches = result.get("matches", [])

            if not matches:
                logger.debug(f"No memories retrieved for user={user_id}.")
                return []

            memories = [match["metadata"]["memory"] for match in matches]

            logger.debug(
                f"[Memory Retrieved] user={user_id} | query='{query}' "
                f"| matches={len(memories)}"
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise AppException(e, sys)