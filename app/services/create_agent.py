import os
import sys
from dotenv import load_dotenv
from langchain.schema import BaseRetriever

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate

from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params

load_dotenv()


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

            # Agent System Prompt

            template = """Answer the following questions as best you can. You have access to the following tools:
            {tools} 
            The tools represent the document(s) provided by the user and must be used as the primary source of truth for answering questions.
            If the user asks a question that CAN be answered using the information found through the tools, you MUST use the tools to retrieve and answer strictly based on the document content.

            If the user asks a question that CANNOT be answered from the document, respond with:
            "The provided document does not contain information about your question. However, from general knowledge..."
            Then continue with a clear, conversational explanation based on your general LLM knowledge.

            Use the following format:
            Question: the input question you must answer Thought: you should always think about what to do 
            Action: the action to take, should be one of [{tool_names}] Action Input: the input to the action 
            Observation: the result of the action ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer Final Answer: the final answer to the original input question

            Begin!

            Question: {input} Thought:{agent_scratchpad}"""

            prompt = PromptTemplate.from_template(template)

            # Create the ReAct Agent
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt,
                # output_parser=StrOutputParser()
            )

            return agent, tools

        except Exception as e:
            self.logger.error(f"Error creating ReAct Agent: {e}")
            raise AppException(e, sys)
