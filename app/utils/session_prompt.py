import sys
from pathlib import Path
from typing import Dict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from app.core.exception import AppException
from app.core.logger import setup_logger
from app.utils.params import load_params

params = load_params('config/params.yaml')
session_prompt_params = params.get("session_prompt_params", {})

log_file_path = session_prompt_params.get("log_file_path", "session_prompt.log")

logger = setup_logger("SessionPromptManager", log_file_path)


class SessionPromptManager:
    """
    Manages chat session histories and prompt loading for conversational RAG systems.
    
    This class handles:
        - Creation and retrieval of per-session chat histories
        - Loading of system prompts from local files with fallback defaults
        - Robust error handling and logging for observability
    
    Typical usage example:
        >>> manager = SessionPromptManager()
        >>> history = manager.get_session_history("user123")
        >>> prompt = manager.load_prompt("app/prompts/default_prompt.txt")
    """

    def __init__(self):
        # In-memory cache for user chat histories
        self._history: Dict[str, BaseChatMessageHistory] = {}
        logger.info("SessionPromptManager initialized successfully.")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve or create a chat history object for the given session ID.

        Args:
            session_id (str): Unique identifier for a user session.

        Returns:
            BaseChatMessageHistory: The chat history associated with the session.
        """
        try:
            if session_id not in self._history:
                self._history[session_id] = ChatMessageHistory()
                logger.info(f"New chat history created for session: {session_id}")
            else:
                logger.debug(f"Existing chat history retrieved for session: {session_id}")

            return self._history[session_id]

        except Exception as e:
            logger.error(f"Failed to retrieve session history for {session_id}: {e}")
            raise AppException(e, sys)

    def load_prompt(self, path: str) -> str:
        """
        Load a system or contextual prompt from a given file path.

        Args:
            path (str): Path to the prompt file.

        Returns:
            str: The prompt text. Returns a default fallback prompt if file not found.
        """
        try:
            prompt_path = Path(path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {path}. Using default prompt.")
                return "You are a helpful assistant. Answer questions based on the provided context."

            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt_text = file.read().strip()
                logger.info(f"Prompt loaded successfully from: {path}")
                return prompt_text

        except Exception as e:
            logger.error(f"Error while loading prompt file {path}: {e}")
            raise AppException(e, sys)
