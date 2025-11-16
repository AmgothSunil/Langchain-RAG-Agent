# LangChain RAG Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-blueviolet)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-FF9900?logo=amazon-aws)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/features/actions)

A **production-ready Conversational RAG (Retrieval-Augmented Generation) Agent** built with modern AI/ML technologies. This system combines FastAPI backend, LangChain ReAct agent framework, Google Gemini LLM, Pinecone vector database, MongoDB storage, and Streamlit frontend—all deployable to AWS ECS Fargate with automated CI/CD pipelines.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

This project implements an intelligent conversational agent that combines multiple technologies to provide context-aware responses with document retrieval capabilities. The system maintains conversation history, semantic understanding, and document embeddings to deliver more accurate and relevant responses than traditional chatbots.

### Use Cases

- **Intelligent Document Q&A Systems**: Upload documents and ask natural language questions
- **Context-Aware Chatbots**: Maintain conversation history with semantic understanding
- **Knowledge Base Assistants**: Combine multiple documents for comprehensive answers
- **Customer Support Automation**: Handle complex queries with document context
- **Internal Knowledge Systems**: Deploy as enterprise solutions on AWS

## Key Features

### 🧠 Multi-Level Memory Architecture

- **Short-term Memory**: Session-based chat history stored in MongoDB with efficient indexing
- **Long-term Memory**: Semantic embeddings and context stored in Pinecone vector database
- **Intelligent Recall**: Automatically retrieves relevant context based on query similarity

### 🤖 ReAct-Based LLM Agent

- **Powered by Google Gemini 2.5 Flash**: Fast, efficient inference with strong reasoning
- **ReAct Framework**: Thought → Action → Observation → Answer cycle for transparent reasoning
- **Tool Integration**: Seamless integration of retrieval and custom tools
- **Dynamic Tool Usage**: Agent automatically decides when to retrieve documents

### 📄 Document Management

- **Upload & Processing**: Automatic document preprocessing and chunking
- **Semantic Embedding**: Convert documents to embeddings for similarity search
- **Vector Retrieval**: Pinecone-based similarity search for relevant content
- **Multi-Format Support**: Handle various document types (PDF, TXT, DOCX)

### 🌐 Production-Ready APIs

- **FastAPI Backend**: High-performance, async-ready REST API
- **Type Safety**: Full Pydantic validation for request/response models
- **Error Handling**: Comprehensive exception handling and logging
- **Health Checks**: Built-in monitoring endpoints

### 🎨 User-Friendly Interface

- **Streamlit UI**: Minimal, responsive chat interface
- **Real-time Interaction**: Live streaming responses
- **Document Upload**: Drag-and-drop document interface
- **Session Management**: Maintain separate conversations per user

### ☁️ Enterprise Deployment

- **Docker Containerization**: Optimized image for production
- **AWS ECS Fargate**: Serverless container orchestration
- **AWS ECR**: Secure container image registry
- **GitHub Actions CI/CD**: Automated testing and deployment
- **Load Balancer**: Distributed traffic management
- **Scalable Infrastructure**: Horizontal scaling for high traffic

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.10+ |
| **Backend Framework** | FastAPI | 0.100+ |
| **Frontend** | Streamlit | 1.28+ |
| **LLM** | Google Gemini 2.5 Flash | Latest |
| **Agent Framework** | LangChain | 0.1+ |
| **Vector Database** | Pinecone | - |
| **Memory Storage** | MongoDB | 5.0+ |
| **Containerization** | Docker | 20.10+ |
| **Container Registry** | AWS ECR | - |
| **Orchestration** | AWS ECS Fargate | - |
| **CI/CD** | GitHub Actions | - |
| **HTTP Client** | HTTPX, Requests | Latest |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Frontend UI          │      External API Clients     │
└──────────┬──────────────────────┴──────────────────┬────────────┘
           │                                         │
           └─────────────────────┬───────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   FastAPI Backend        │
                    │  - /chat                 │
                    │  - /upload               │
                    │  - /health               │
                    └─────────┬────────────────┘
                              │
                ┌─────────────┼──────────────┐
                │             │              │
       ┌────────▼──────┐ ┌────▼────────┐ ┌─▼────────────┐
       │  Agent Layer  │ │  Retriever  │ │  LLM Layer   │
       │  (LangChain   │ │   (RAG)     │ │  (Gemini)    │
       │   ReAct)      │ │             │ │              │
       └────────┬──────┘ └────┬────────┘ └──────────────┘
                │             │
       ┌────────▼──────────────▼──────────────┐
       │         Memory Management            │
       ├─────────────────┬────────────────────┤
       │   MongoDB       │   Pinecone         │
       │  (Short-term)   │  (Long-term)       │
       └─────────────────┴────────────────────┘
```

## Prerequisites

### Required

- **Python 3.10 or higher**
- **Git** for version control
- **Docker** (for containerization)
- **Docker Desktop** or Docker daemon running

### External Services

- **Google Cloud Project** with Gemini API enabled
- **Pinecone Account** with index created
- **MongoDB Atlas or Local MongoDB** instance
- **AWS Account** (for deployment to ECS)

### API Keys Required

- `GOOGLE_API_KEY`: Google Gemini API key
- `PINECONE_API_KEY`: Pinecone API key
- AWS credentials configured locally (for deployment)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmgothSunil/Langchain-RAG-Agent.git
cd Langchain-RAG-Agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (faster alternative)
uv sync
```

### 4. Verify Installation

```bash
python -c "import fastapi, langchain, streamlit; print('All dependencies installed')"
```

## Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Update `.env` with Your Credentials

```bash
LANGCHAIN_API_KEY="lsv2... Your Langchain API KEY"
LANGCHAIN_PROJECT="Agentic-RAG-ChatBot--->Your Project Name"
HUGGINGFACE_API_TOKEN="hf_ Your Hugging Face Token"
GOOGLE_API_KEY="AIza_ Your Google API Key"
PINECONE_API_KEY="pcsk_ Your Pinecone API Key"
PINECONE_VECTORS_INDEX_NAME="agent-vector-store  ---> Your Pinecone Vectors Index Name"
PINECONE_MEMORY_INDEX_NAME="agent-memory-store   ---> Your Pinecone Memory Index Name"
MONGO_URI="mongodb+srv: Your Mangodb URI Here"
MONGO_DB_NAME="agent_memory_db  ---> Your Mandodb Database Name"
MONGO_COLLECTION="agent_history ---> Your Table Collection Name"
USER_AGENT="LangChain-RAG-Agent/1.0   ---> Your  User Agent"
ENVIRONMENT="production   ---> Your Envirement"
DEBUG="True   ---> Your Debug Value"
```

### 3. Verify Configuration

```bash
python -c "from app.utils.params import load_config; config = load_config(); print('✅ Configuration loaded successfully')"
```

## Running Locally

### Backend (FastAPI)

```bash
# Start the FastAPI server
uv run app/api/fastapi_app.py
```

The backend API will be available at:
- **Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Frontend (Streamlit)

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Run Streamlit app
streamlit run streamlit_app.py
```

The UI will be available at:
- **Frontend**: `http://localhost:8501`

### Verify Both Services

```bash
# Check backend health
curl http://localhost:8000/health

# Response should be:
# {"status":"healthy","timestamp":"2024-01-15T10:30:00Z"}
```

## API Documentation

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, the API is open. For production, consider implementing API key authentication or OAuth2.

### Endpoints

#### 1. Chat Endpoint

**POST** `/chat`

Send a message to the RAG agent and receive a response.

**Request Body**:

```json
{
  "user_id": "user123",
  "session_id": "session456",
  "message": "What is the main topic in the uploaded documents?",
  "include_sources": true
}
```

**Response**:

```json
{
  "response": "Based on the documents, the main topic is...",
  "reasoning": "Thought: I need to search the documents...",
  "sources": [
    {
      "document": "file.pdf",
      "page": 1,
      "excerpt": "..."
    }
  ],
  "session_id": "session456",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid request format
- `500`: Server error

#### 2. Upload Document Endpoint

**POST** `/upload`

Upload a document to be indexed and made available for retrieval.

**Request**:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "user_id=user123"
```

**Response**:

```json
{
  "filename": "document.pdf",
  "file_id": "file_abc123",
  "chunks": 45,
  "embedding_status": "completed",
  "message": "Document successfully uploaded and indexed"
}
```

**Supported Formats**: PDF, TXT, DOCX, CSV

#### 3. Health Check Endpoint

**GET** `/health`

Monitor service health and dependencies.

**Response**:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "api": "healthy",
    "mongodb": "connected",
    "pinecone": "connected",
    "gemini": "ready"
  }
}
```

### Example: Complete Chat Flow

```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@research_paper.pdf" \
  -F "user_id=user123"

# 2. Send a chat message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "session456",
    "message": "Summarize the key findings from the paper",
    "include_sources": true
  }'
```

## Deployment

### Docker Setup

#### Build Docker Image

```bash
# Build the image
docker build -t rag-agent:latest .

# Verify the build
docker images | grep rag-agent
```

#### Run Docker Container Locally

```bash
# Create .env file with your credentials
docker run -p 8000:8000 \
  --env-file .env \
  --name rag-agent-container \
  rag-agent:latest
```

#### Stop Container

```bash
docker stop rag-agent-container
docker rm rag-agent-container
```

### AWS ECS Fargate Deployment

#### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- ECR repository created

#### Step 1: Push Image to ECR

```bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag the image
docker tag rag-agent:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest

# Push to ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest
```

#### Step 2: Update Task Definition

Edit `ecs-task-def.json`:

```json
{
  "family": "rag-agent-task",
  "containerDefinitions": [
    {
      "name": "rag-agent",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "GOOGLE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:google-api-key"
        }
      ]
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "512",
  "memory": "1024"
}
```

#### Step 3: Deploy to ECS

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json

# Update service
aws ecs update-service \
  --cluster rag-agent-cluster \
  --service rag-agent-service \
  --force-new-deployment
```

### GitHub Actions CI/CD

The project includes automated CI/CD via `.github/workflows/deploy.yaml`:

**Triggers on**:
- Push to `main` branch
- Manual workflow dispatch

**Workflow Steps**:
1. Run tests with pytest
2. Build Docker image
3. Push to AWS ECR
4. Update ECS task definition
5. Deploy to ECS Fargate
6. Verify deployment health

**Secrets Required in GitHub**:

```
AWS_ACCOUNT_ID
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION (e.g., us-east-1)
```

Add these in Settings → Secrets and variables → Actions.

### Manual Deployment

```bash
# Full deployment workflow
git add .
git commit -m "Deploy: Update agent logic"
git push origin main

# GitHub Actions automatically triggers deployment
# Monitor in Actions tab of your repository
```

## Project Structure

```
LANGCHAIN-RAG-AGENT/
├── .github/
│   └── workflows/
│       └── deploy.yaml                 # GitHub Actions CI/CD pipeline
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── fastapi_app.py             # Main FastAPI application
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exception.py               # Custom exception classes
│   │   └── logger.py                  # Logging configuration
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── mongo_database.py          # MongoDB operations
│   │   └── pinecone_memory_db.py      # Pinecone vector store
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_bot.py               # Agent execution logic
│   │   ├── create_agent.py            # Agent initialization
│   │   └── preprocess.py              # Document preprocessing
│   │
│   ├── prompts/
│   │   └── create_agent_prompt.txt    # ReAct agent system prompt
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── params.py                  # Configuration management
│   │   └── prompt.py                  # Prompt utilities
│   │
│   └── config/
│       └── params.yaml                # Configuration file
│
├── frontend/
│   ├── __init__.py
│   ├── requirements.txt               # Streamlit dependencies
│   └── streamlit_app.py              # Streamlit UI application
│
├── .dockerignore                      # Docker ingnore rules
├── .env.example                       # Example environment variables
├── .gitignore                         # Git ignore rules
├── .python-version                    # Project implemented python version
├── Dockerfile                         # Docker image configuration
└── LICENSE                           # MIT License
├── README.md                         # This file
├── ecs-task-def.json                 # AWS ECS task definition
├── pyproject.toml                    # Project metadata
├── requirements.txt                  # Python dependencies
├── uv.lock                           # Dependency lock file (uv)

```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `app/api/` | FastAPI endpoints and route handlers |
| `app/core/` | Core utilities (logging, exceptions) |
| `app/db/` | Database connection and operations |
| `app/services/` | Business logic (agent, preprocessing) |
| `app/utils/` | Helper functions and configuration |
| `app/prompts/` | LLM system prompts and templates |
| `frontend/` | Streamlit UI components |

## How It Works

### Conversation Flow

```
1. User Input
   ↓
2. Fetch Short-term Memory (MongoDB)
   ├─ Previous messages in session
   └─ Conversation context
   ↓
3. Retrieve Long-term Memory (Pinecone)
   ├─ Semantic similarity search
   ├─ Document embeddings lookup
   └─ Relevant context retrieval
   ↓
4. Build Complete Prompt
   ├─ System instructions
   ├─ Conversation history
   ├─ Retrieved context
   └─ Current user message
   ↓
5. ReAct Agent Execution
   ├─ Thought: Internal reasoning
   ├─ Action: Tool selection and use
   ├─ Observation: Tool results
   └─ Final Answer: Deterministic response
   ↓
6. Store Response
   ├─ MongoDB: Chat history
   ├─ Pinecone: Response embeddings (optional)
   └─ Logs: Audit trail
   ↓
7. Return to User
   └─ Response with sources and reasoning
```

### Memory Architecture

**Short-term Memory (MongoDB)**:
- Stores every message in the conversation
- Indexed by `session_id` and `user_id`
- TTL-based cleanup for old sessions (configurable)
- Fast retrieval for recent context

**Long-term Memory (Pinecone)**:
- Document embeddings using Gemini Embedding API
- Semantic search for relevant documents
- Metadata tags for filtering
- Vector dimension: 768 (Gemini embeddings)

### ReAct Agent Cycle

1. **Thought**: Agent thinks about the problem
2. **Action**: Agent selects a tool (retriever, calculator, etc.)
3. **Observation**: Tool returns results
4. **Thought/Action loop**: May repeat multiple times
5. **Final Answer**: Agent provides the conclusion


## Troubleshooting

### Common Issues

#### 1. MongoDB Connection Error

```
Error: Unable to connect to MongoDB
```

**Solution**:
- Verify MongoDB is running: `mongosh` or check Atlas connection
- Check `MONGO_URI` in `.env`
- Ensure MongoDB network access rules allow your IP

#### 2. Pinecone Index Not Found

```
Error: Index 'agent-memory-store' not found
```

**Solution**:
- Create index in Pinecone dashboard
- Verify `PINECONE_INDEX_NAME` matches exactly
- Check Pinecone API key permissions

#### 3. Google API Key Invalid

```
Error: Invalid API key provided to Google Gemini
```

**Solution**:
- Regenerate key in Google Cloud Console
- Enable Generative Language API
- Verify no extra spaces in `.env`

#### 4. Port Already in Use

```
Error: Address already in use: 0.0.0.0:8000
```

**Solution**:
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app.api.fastapi_app:app --port 8001
```

#### 5. Streamlit Connection Refused

```
Error: Connection refused to localhost:8000
```

**Solution**:
- Ensure FastAPI backend is running
- Check backend is accessible: `curl http://localhost:8000/health`
- Verify no firewall blocking

### Debug Mode

Enable verbose logging:

```bash
export LOG_LEVEL=DEBUG
uv run app/api/fastapi_app.py
```

## Performance Optimization

### Caching Strategies

- Implement Redis for session caching
- Cache embeddings lookup results
- Batch document indexing

### Scaling Considerations

- Use horizontal scaling on ECS with load balancer
- Optimize MongoDB indexing
- Implement connection pooling
- Use async I/O throughout

### Monitoring

- Set up CloudWatch logs for AWS deployment
- Monitor token usage on Gemini API
- Track Pinecone query latency
- Monitor MongoDB connection pool

## Contributing

Contributions are welcome! Please follow these guidelines:

### 1. Fork the Repository

```bash
git clone https://github.com/AmgothSunil/Langchain-RAG-Agent.git
cd Langchain-RAG-Agent
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow PEP 8 style guide
- Add type hints to functions
- Include docstrings
- Write tests for new features

### 4. Commit and Push

```bash
git add .
git commit -m "feat: Add new feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Submit a PR with:
- Clear description of changes
- Reference any related issues
- Screenshots/examples if applicable

### Code Quality Standards

- **Linting**: Use `black` and `flake8`
- **Type Checking**: Mypy for static type analysis
- **Testing**: Minimum 80% code coverage
- **Documentation**: Docstrings for all functions

```bash
# Format code
black app/ frontend/

# Check linting
flake8 app/ frontend/

# Type checking
mypy app/
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

### License Summary

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Use in patents

Conditions:
- ⚠️ Include license and copyright notice

Limitations:
- ❌ Liability
- ❌ Warranty

## Support

### Getting Help

- **Documentation**: Check `docs/` folder for detailed guides
- **Issues**: Open an issue on GitHub for bugs or features
- **Discussions**: Use GitHub Discussions for questions

### Reporting Issues

When reporting bugs, include:
- Python version and OS
- Error message and traceback
- Steps to reproduce
- Expected vs actual behavior

### Example Issue

```markdown
### Bug Report: Chat endpoint returns 500 error

**Python Version**: 3.10.5
**OS**: Ubuntu 22.04

**Description**:
When uploading a document and sending chat message, endpoint returns 500 error.

**Steps to Reproduce**:
1. Upload PDF via /upload endpoint
2. Send message via /chat endpoint
3. Observe error

**Error Message**:
```
AttributeError: 'NoneType' object has no attribute 'split'
```

**Expected**: Should return chat response with sources
```

### Community

- Join discussions for feature requests
- Share projects built with this framework
- Contribute improvements and fixes

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://www.langchain.com/) - LLM framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) - Language model
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Streamlit](https://streamlit.io/) - Web UI framework
- [MongoDB](https://www.mongodb.com/) - Document database


**Questions?** Open an issue or start a discussion on GitHub.

**Star us** ⭐ if this project helps you!

---

Made with ❤️ for production-ready AI applications
