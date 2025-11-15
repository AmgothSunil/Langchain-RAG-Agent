import os
import uuid
import requests
import streamlit as st
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

UPLOAD_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/upload-docs"
CHAT_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/chat"

# Streamlit Page Configuration
st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("End-to-End Conversational RAG Chatbot")
st.caption("Powered by FastAPI • Pinecone • MongoDB • LangChain")

# Session Management
def initialize_session():
    """Initialize or restore chat session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.uploaded = False
        st.info("New chat session started!")

def clear_session():
    """Start a fresh chat session."""
    st.session_state.clear()
    st.rerun()

initialize_session()

# Helper Functions
def upload_documents(files: List, urls:str, session_id: str):
    """Upload documents to backend for vector storage."""
    try:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]

        if not files and not url_list:
            st.warning("Please select files or enter URLs before processing.")
            return False

        files_payload = [("files", (f.name, f, f.type)) for f in files]
        data_payload = {"session_id": session_id}

        if url_list:
            data_payload['urls'] = url_list

        with st.spinner("Processing sources... This may take a moment."):
            response = requests.post(UPLOAD_ENDPOINT, files=files_payload, data=data_payload, timeout=300)

        if response.status_code == 200:
            st.success("Sources processed and embedded successfully.")
            st.session_state.uploaded = True
            return True
        else:
            st.error(f"Processing failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return False


def chat_with_bot(question: str, session_id: str):
    """Send chat query to backend and retrieve AI response."""
    try:
        payload = {"question": question, "session_id": session_id}
        with st.spinner("Thinking..."):
            response = requests.post(CHAT_ENDPOINT, data=payload, timeout=90)

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response received.")
        elif response.status_code == 400:
            st.warning(response.json().get("detail", "Please upload documents first."))
        else:
            st.error(f"Server error [{response.status_code}]: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")
    return None


def display_chat_history():
    """Render the chat history UI."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# Sidebar: Upload and Session Controls
with st.sidebar:
    st.header("📂 Add to Knowledge Base")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF/TXT files to build the knowledge base."
    )

    st.markdown("---")
    pasted_urls = st.text_area(
        "Or paste URLs (one per line)",
        height=150,
        placeholder="https://example.com/page1\nhttps://blog.example.com/post2"
    )

    # Check if there is anything to process 
    if uploaded_files or pasted_urls.strip():
        if st.button("Process Sources"):
            upload_documents(uploaded_files, pasted_urls, st.session_state.session_id)

    st.markdown("---")
    st.subheader("⚙️ Session Controls")
    st.code(f"Session ID: {st.session_state.session_id}", language="text")

    if st.button("Start New Chat"):
        clear_session()

    st.markdown("---")
    st.caption("Backend: FastAPI | Vector Store: Pinecone | DB: MongoDB")

# Chat Interface
st.subheader("💭 Chat with Your Documents")
st.write("Ask questions based on your uploaded files using Retrieval-Augmented Generation (RAG).")

display_chat_history()

if question := st.chat_input("Ask something about your documents..."):
    if not st.session_state.uploaded:
        st.warning("Please upload and process sources before starting a chat.")
    else:
        # Append user message and display it immediately
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get and display bot response
        response = chat_with_bot(question, st.session_state.session_id)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
