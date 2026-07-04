"""Streamlit chatbot application using AWS Bedrock Knowledge Base.

All sensitive configuration (Knowledge Base ID, DB credentials) is fetched from AWS Secrets Manager.
No hardcoded values.
"""
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from config.settings import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    KNOWLEDGE_BASE_SECRET_ARN,
)
from src.bedrock_client import BedrockClient
from src.embedding_reranker import is_relevant_context, rerank_by_embedding

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_configuration() -> bool:
    """Validate that all required configuration is present.
    
    Returns:
        True if configuration is valid, False otherwise.
    """
    errors = []
    
    if not AWS_REGION:
        errors.append("AWS_REGION not set")
    
    if not KNOWLEDGE_BASE_SECRET_ARN:
        errors.append("KNOWLEDGE_BASE_SECRET_ARN not set")
    
    if not BEDROCK_MODEL_ID:
        errors.append("BEDROCK_MODEL_ID not set")
    
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False
    
    return True


@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client.
    
    Knowledge Base ID is fetched from AWS Secrets Manager at initialization.
    """
    try:
        logger.info("Initializing Bedrock client...")
        client = BedrockClient(
            kb_id_secret_arn=KNOWLEDGE_BASE_SECRET_ARN,
            model_id=BEDROCK_MODEL_ID,
            region=AWS_REGION,
        )
        logger.info("Bedrock client initialized successfully")
        return client
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        logger.error(f"Bedrock configuration error: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {e}")
        logger.exception("Bedrock initialization error")
        return None


# Page configuration
st.set_page_config(
    page_title="Bedrock Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Validate configuration on startup
if not validate_configuration():
    st.error(
        "❌ Configuration Error\n\n"
        "Please set required environment variables:\n"
        "- AWS_REGION\n"
        "- KNOWLEDGE_BASE_SECRET_ARN\n"
        "- BEDROCK_MODEL_ID\n\n"
        "For ECS, set these in the task definition environment variables."
    )
    st.stop()

# Configuration sidebar (display only)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # st.write(f"**AWS Region:** {AWS_REGION}")
    # st.write(f"**Model:** {BEDROCK_MODEL_ID}")
    st.write("**Data Source:** AWS Bedrock Knowledge Base")
    st.write("✓ Knowledge Base ID from AWS Secrets Manager")
    st.write("✓ Embeddings stored in RDS (managed by Bedrock)")
    
    st.markdown(
        """
        ---
        **Security Notes:**
        - All credentials are fetched from AWS Secrets Manager
        - No hardcoded secrets in code or configuration
        - For ECS Fargate, credentials are retrieved using task role
        """
    )

st.title("🤖 Bedrock Chatbot")
st.write("Ask questions about your knowledge base")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.json(source)

# Chat input
prompt = st.chat_input("Ask a question about your knowledge base")
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Retrieving context and generating answer..."):
        try:
            bedrock_client = get_bedrock_client()
            if bedrock_client is None:
                st.error("❌ Bedrock client not initialized. Check configuration and logs.")
                logger.error("Bedrock client initialization failed")
            else:
                # Retrieve context from Bedrock Knowledge Base
                logger.debug(f"Retrieving context for: {prompt[:50]}...")
                sources = bedrock_client.retrieve_context(prompt, top_k=15)

                if not sources:
                    answer = "I don't know — no evidence in knowledge base."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                else:
                    # Rerank using embeddings (single implementation - no duplication)
                    logger.debug(f"Reranking {len(sources)} results...")
                    selected_chunks = rerank_by_embedding(sources, prompt, top_k=8)

                    # Check relevance
                    if not selected_chunks or not is_relevant_context(prompt, selected_chunks):
                        answer = "I don't know — no evidence in knowledge base."
                    else:
                        # Generate answer with Bedrock
                        logger.debug("Generating answer with Bedrock...")
                        answer = bedrock_client.generate_answer(prompt, selected_chunks)

                    # Add assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                    # Display response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if sources:
                            with st.expander("📄 Sources"):
                                for source in sources:
                                    st.json(source)

        except Exception as e:
            logger.exception("Error processing query")
            st.error(f"❌ Error: {str(e)}")
