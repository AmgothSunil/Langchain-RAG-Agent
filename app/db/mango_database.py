import os
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from app.utils.params import load_params
from app.core.logger import setup_logger
from app.core.exception import AppException

load_dotenv()

params = load_params("config/params.yaml")
mongo_params = params.get("mangodb_params", {})
log_file_path = mongo_params.get("log_file_path", "mongodb.log")

logger = setup_logger(name="MongoDBManager", log_file_name=log_file_path)


class AsyncMongoDatabase:
    """Async class for handling MongoDB operations."""

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.database_name = os.getenv("MONGO_DB_NAME", "agent_memory_db")
        self.collection_name = os.getenv("MONGO_COLLECTION", "agent_history")

        if not self.mongo_uri:
            raise AppException("MONGO_URI is not set in environment variables.", sys)

        try:
            logger.info("Initializing async MongoDB client...")
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
        except Exception as e:
            logger.error("Failed to initialize MongoDB: %s", e)
            raise AppException(e, sys)

    async def save_chat(self, session_id: str, user_input: str, bot_response: str):
        """Asynchronously save a chat interaction to MongoDB."""
        try:
            document = {
                "session_id": session_id,
                "user_input": user_input,
                "chatbot_response": bot_response,
            }
            await self.collection.insert_one(document)
            logger.debug("Chat saved to MongoDB successfully.")
        except Exception as e:
            logger.error("Error inserting chat document: %s", e)
            raise AppException(e, sys)

    async def fetch_recent_chats(self, session_id: str, limit: int = 5) -> str:
        """Fetch recent chats asynchronously."""
        try:
            cursor = self.collection.find(
                {"session_id": session_id},
                {"_id": 0, "user_input": 1, "chatbot_response": 1}
            ).sort("_id", -1).limit(limit)

            chats = [chat async for chat in cursor]
            chats.reverse()

            if not chats:
                logger.info("No previous chats found for session_id: %s", session_id)
                return ""

            context = "\n".join(
                [f"User: {chat['user_input']}\nAssistant: {chat['chatbot_response']}" for chat in chats]
            )
            return context

        except Exception as e:
            logger.error("Error fetching chat history: %s", e)
            return ""

    async def close_connection(self):
        """Close MongoDB connection."""
        try:
            self.client.close()
            logger.info("MongoDB connection closed.")
        except Exception as e:
            logger.error("Error closing MongoDB connection: %s", e)