import os
import uvicorn
import tempfile
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from dotenv import load_dotenv

from app.core.logger import setup_logger
from app.core.exception import AppException
from app.utils.params import load_params
from app.services.preprocess import DocumentPreprocessor
from app.services.create_agent import CreateAgent
from app.services.agent_bot import AgentChatbot
from app.db.mango_database import AsyncMongoDatabase


# Load environment
load_dotenv()

# LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


params = load_params("config/params.yaml")
server_params = params.get("server_params", {})
log_file_path = server_params.get("log_file_path", "server.log")

logger = setup_logger("ConversationalRAGServer", log_file_path)

mango_db = AsyncMongoDatabase()
document_preprocessor = DocumentPreprocessor()

retriever_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server startup & shutdown."""
    logger.info("Starting Conversational RAG Server...")
    yield
    logger.info("Shutting down Conversational RAG Server...")
    await mango_db.close_connection()


app = FastAPI(
    title="Conversational RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Conversational RAG API running"}


@app.post("/upload-docs", status_code=HTTP_200_OK)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(None),
    urls: Optional[List[str]] = Form(None)
):
    """Upload documents and build retriever."""
    try:
        if not files and not urls:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Provide at least one document or URL."
            )

        temp_paths = []

        if files:
            for file in files:
                suffix = os.path.splitext(file.filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(await file.read())
                    temp_paths.append(temp.name)

        all_sources = temp_paths + (urls or [])

        docs = document_preprocessor.load_documents(all_sources)
        retriever = document_preprocessor.build_retriever(docs, session_id=session_id)

        retriever_cache[session_id] = retriever

        # Cleanup temp files
        for path in temp_paths:
            os.unlink(path)

        return {"message": "Documents processed successfully"}

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing uploaded documents."
        )


@app.post("/chat", status_code=HTTP_200_OK)
async def chat_with_bot(
    session_id: str = Form(...),
    question: str = Form(...)
):
    """Main chatbot endpoint."""
    try:
        retriever = retriever_cache.get(session_id)

        if retriever is None:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="You must upload documents first."
            )

        # Build agent
        agent_builder = CreateAgent()
        agent, tools = agent_builder.create_agent(retriever)

        # Chat
        bot = AgentChatbot(question=question, session_id=session_id)
        response = await bot.chatbot(agent, tools)

        return {
            "session_id": session_id,
            "question": question,
            "response": response
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating chatbot response."
        )


if __name__ == "__main__":
    uvicorn.run("app.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
