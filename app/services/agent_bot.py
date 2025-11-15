import sys
from typing import Optional, List

from langchain.agents import AgentExecutor, Tool
from langchain_core.runnables import Runnable

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params
from app.db.mango_database import AsyncMongoDatabase
from app.db.pinecone_memory_db import PineconeMemory


params = load_params("config/params.yaml")
rag_chatbot_params = params.get("rag_chatbot_params", {})
log_file_path = rag_chatbot_params.get("log_file_path", "rag_chatbot.log")
chat_history_limit = rag_chatbot_params.get("chat_history_limit", 5)

logger = setup_logger("RAGChatbot", log_file_path)

mango_db = AsyncMongoDatabase()
pc_memory = PineconeMemory()


class AgentChatbot:
    """
    Orchestrates the complete conversational RAG flow:
        - Retrieves long-term semantic memory from Pinecone
        - Retrieves short-term chat history from Redis
        - Invokes the ReAct agent with memory injected
        - Saves full chat logs to Redis
        - Updates long-term memory after response
    """

    def __init__(self, question: str, session_id: str):
        self.question = question
        self.session_id = session_id

    async def chatbot(self, agent: Runnable, tools: List[Tool]) -> Optional[str]:
        try:
            logger.info(f"Chat session started | session_id={self.session_id}")

            # 1️⃣ Retrieve past short-term history
            history = await mango_db.fetch_recent_chats(
                self.session_id, chat_history_limit
            )

            # 2️⃣ Retrieve long-term semantic memory
            semantic_memory = pc_memory.retrieve_memory(
                self.session_id, self.question
            )

            # 3️⃣ Build final agent input
            final_prompt = f"""
                            Short-Term Conversation History:
                            {history}

                            Long-Term Semantic Memory:
                            {semantic_memory}

                            User Query:
                            {self.question}
                            """

            # 4️⃣ Create AgentExecutor
            executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                handle_parsing_errors=True,
                verbose=True
            )

            # 5️⃣ Call the agent
            result = executor.invoke({
                "input": final_prompt,
                # "agent_scratchpad": []    
            })

            response_text = result["output"]  # ReAct always returns "output"

            # 6️⃣ Save chat to Redis
            await mango_db.save_chat(
                self.session_id,
                self.question,
                response_text
            )

            # 7️⃣ Store NEW memory (post-response)
            pc_memory.store_memory(self.session_id, self.question)

            logger.info(f"Chat session completed | session_id={self.session_id}")
            return response_text

        except Exception as e:
            logger.error(f"Unexpected chatbot error: {e}", exc_info=True)
            raise AppException("Chatbot failed.", sys)
