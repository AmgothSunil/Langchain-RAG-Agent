import os
import sys
from dotenv import load_dotenv
from langchain.schema import BaseRetriever

from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params
from app.utils.prompt import PromptManager

load_dotenv()

prompt_manager = PromptManager()

class CreateAgent:
    """
    Factory class for building a LangChain ReAct Agent integrated with:
    
    - Gemini LLM (Google GenAI)
    - Document retrieval (Pinecone, Chroma, etc.)
    - RAG-style document lookup through a retriever tool

    This class is responsible for constructing an agent that can:
      • use uploaded documents or URLs via a retriever tool  
      • perform reasoning steps through ReAct  
      • answer questions based on retrieved or inferred information  

    Typical usage:
        >>> agent_builder = CreateAgent()
        >>> agent = agent_builder.create_agent(retriever)
        >>> response = agent.invoke({"input": "Summarize the document"})
    """

    def __init__(self):
        """
        Initialize the LLM and validate required environment variables.

        Args:
            uploaded_files (optional): Reserved for future use (document preprocessing).
        """
        try:
            params = load_params("config/params.yaml")
            agent_params = params.get("create_agent_params", {})
            log_file_path = agent_params.get("log_file_path", "agent.log")

            self.logger = setup_logger("CreateAgent", log_file_path)
            self.logger.debug("Initializing CreateAgent...")

            # API Key Validation
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise AppException("GOOGLE_API_KEY missing in environment variables.", sys)

            # Initialize Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=self.google_api_key,
                temperature=0.2,
            )

            self.logger.info("Gemini LLM initialized successfully.")

        except Exception as e:
            self.logger.error(f"Failed to initialize CreateAgent: {e}")
            raise AppException(e, sys)

    def create_agent(self, retriever: BaseRetriever):
        """
        Build a ReAct-style LangChain Agent equipped with:
        
        - A retriever tool (backed by Pinecone)
        - Google Gemini LLM (via ChatGoogleGenerativeAI)
        - Structured ReAct prompt template
        - Safe output parsing
        
        Args:
            retriever (BaseRetriever): A LangChain retriever object for RAG operations.
        
        Returns:
            AgentExecutor: A fully constructed LangChain Agent ready to invoke().
        
        Example:
            >>> agent = create_agent(retriever)
            >>> result = agent.invoke({"input": "Explain section 2 of the PDF"})
        """
        try:
            self.logger.info("Creating ReAct Agent with retriever tool...")

            # Convert retriever into a tool
            retriever_tool = create_retriever_tool(
                retriever,
                name="retriever_tool",
                description="Use this tool to search and retrieve information from uploaded documents."
            )

            tools = [retriever_tool]

            # Load prompt
            template = prompt_manager.load_prompt("app/prompts/create_agent_prompt.txt")

            prompt = PromptTemplate.from_template(template)

            # Create the ReAct Agent
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )

            return agent, tools

        except Exception as e:
            self.logger.error(f"Error creating ReAct Agent: {e}")
            raise AppException(e, sys)
