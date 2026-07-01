import os
import streamlit as st
from src.chatbot import Chatbot

st.set_page_config(page_title="Bedrock Chatbot", page_icon="🤖")
st.title("Bedrock Knowledge Base Chatbot")
st.write("Ask questions about your Amazon Bedrock knowledge base using Amazon Nova Micro.")

# Debug section
with st.sidebar:
    st.header("Debug Info")
    st.write(f"AWS Profile: {os.getenv('AWS_PROFILE', 'default')}")
    st.write(f"Knowledge Base ID: {os.getenv('KNOWLEDGE_BASE_ID', 'Not set')}")
    st.write(f"AWS Region: {os.getenv('AWS_REGION', 'Not set')}")
    st.write(f"Model ID: {os.getenv('BEDROCK_MODEL_ID', 'Not set')}")

with st.sidebar:
    st.header("Configuration")
    kb_id = st.text_input("Knowledge Base ID", value=os.getenv("KNOWLEDGE_BASE_ID", ""))
    model_id = st.text_input("Model ID", value=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0"))
    region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "us-east-1"))
    if st.button("Apply settings"):
        os.environ["KNOWLEDGE_BASE_ID"] = kb_id
        os.environ["BEDROCK_MODEL_ID"] = model_id
        os.environ["AWS_REGION"] = region
        # Force recreate the chatbot with new settings
        if "chatbot" in st.session_state:
            del st.session_state.chatbot
        st.success("Settings updated - refresh the page")

    st.markdown("""
    These values come from .env file:
    - KNOWLEDGE_BASE_ID
    - BEDROCK_MODEL_ID
    - AWS_REGION
    - AWS_PROFILE
    """)

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = Chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

prompt = st.chat_input("Ask a question about your knowledge base")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            result = st.session_state.chatbot.ask(prompt)
        except Exception as e:
            result = {"answer": f"Error: {str(e)}", "sources": []}

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(source)
