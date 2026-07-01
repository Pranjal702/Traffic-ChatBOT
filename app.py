import os
import re
import streamlit as st
import boto3
import numpy as np
from botocore.exceptions import ClientError, BotoCoreError
import logging
import configparser
from src.embeddings import embed_texts
try:
    import src.embedding_reranker as embedding_reranker
except Exception:
    embedding_reranker = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_aws_credentials():
    """Use ECS Task Role - no credentials file needed."""
    return None

def get_bedrock_clients():
    """Create Bedrock clients using ECS Task Role (automatic in Fargate)."""
    region = os.getenv("AWS_REGION", "us-east-1")
    logger.debug(f"Creating Bedrock clients for region: {region}")
    
    try:
        session = boto3.Session(region_name=region)
        bedrock_runtime = session.client("bedrock-runtime", region_name=region)
        agent_runtime = session.client("bedrock-agent-runtime", region_name=region)
        return bedrock_runtime, agent_runtime
    except Exception as e:
        logger.error(f"Failed to create Bedrock clients: {e}")
        raise

def retrieve_context(query: str, kb_id: str):
    """Retrieve context from knowledge base and chunk longer passages."""
    logger.debug(f"Retrieving context for KB: {kb_id}, Query: {query}")
    _, agent_runtime = get_bedrock_clients()
    try:
        response = agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
        )
        retrieval_results = response.get("retrievalResults", [])
        chunks = []
        for doc_idx, item in enumerate(retrieval_results[:15], start=1):
            content = item.get("content", {})
            text = content.get("text", "")
            source = item.get("location", {})
            score = item.get("score") or item.get("similarity") or 0.0
            doc_id = f"SRC_{doc_idx}"
            text_chunks = chunk_text(text, chunk_size=900, overlap=200)
            for chunk_idx, chunk in enumerate(text_chunks, start=1):
                chunks.append({
                    "id": f"{doc_id}-{chunk_idx}",
                    "text": chunk,
                    "source": {"docId": doc_id, "location": source},
                    "score": score,
                })
        logger.debug(f"Retrieved {len(chunks)} chunked passages from {len(retrieval_results[:15])} documents")
        return chunks
    except (ClientError, BotoCoreError) as exc:
        logger.error(f"Retrieve failed: {exc}")
        return [{"id": "SRC_ERR", "text": f"Knowledge base lookup failed: {exc}", "source": {"sourceType": "Error"}, "score": 0.0}]

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, 0)
    return chunks


def get_source_label(location):
    if not location:
        return "unknown"
    if isinstance(location, dict):
        inner = location.get("location", location)
        if isinstance(inner, dict):
            return (
                inner.get("s3Location", {}).get("uri")
                or inner.get("uri")
                or inner.get("s3Path")
                or inner.get("sourceId")
                or inner.get("docId")
                or str(inner)
            )
        return str(inner)
    return str(location)


def rerank_with_embeddings(query: str, context_chunks, top_n: int = 6):
    if not context_chunks:
        return []

    texts = [chunk.get("text", "") for chunk in context_chunks]
    query_embedding = embed_texts([query])[0]
    chunk_embeddings = embed_texts(texts)

    def cosine_similarity(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (denom + 1e-10))

    scored = []
    for chunk, emb in zip(context_chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        chunk_with_score = dict(chunk)
        chunk_with_score["emb_score"] = score
        scored.append(chunk_with_score)

    ranked = sorted(scored, key=lambda item: item.get("emb_score", 0.0), reverse=True)
    return ranked[:top_n]


def generate_answer(query: str, context_chunks, model_id: str):
    bedrock_runtime, _ = get_bedrock_clients()

    # Prefer the new local embedding reranker module when available.
    if embedding_reranker is not None:
        try:
            selected_chunks = embedding_reranker.rerank_by_embedding(context_chunks, query, top_k=8)
        except Exception:
            selected_chunks = rerank_with_embeddings(query, context_chunks, top_n=8)
    else:
        selected_chunks = rerank_with_embeddings(query, context_chunks, top_n=8)
    if not selected_chunks or not embedding_reranker.is_relevant_context(query, selected_chunks):
        return "I don't know — no evidence in knowledge base."

    passage_texts = []
    for chunk in selected_chunks:
        source_label = get_source_label(chunk.get("source"))
        passage_texts.append(
            f"[{chunk['id']}] Source: {source_label}\n{chunk.get('text', '')[:1500]}"
        )

    prompt = (
        "You are a bilingual assistant. Answer in the same language as the question using ONLY the passages below. "
        "Cite sources inline with the source ids (for example [SRC_1]). "
        "If the answer is not present in the passages, respond exactly: 'I don't know — no evidence in knowledge base.' "
        "Keep the answer concise, factual, and based only on the provided context.\n\n"
        f"QUESTION:\n{query}\n\nPASSAGES:\n" + "\n\n".join(passage_texts) + "\n\nAnswer:"
    )

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 350, "temperature": 0.0, "topP": 0.9},
        )
        output = response.get("output", {}).get("message", {}).get("content", [])
        if output:
            return output[0].get("text", "")
        return "No response was returned by Bedrock."
    except (ClientError, BotoCoreError) as exc:
        logger.error(f"Converse failed: {exc}")
        try:
            error_body = getattr(exc, 'response', None)
            return f"Bedrock call failed: {exc} | response: {error_body}"
        except Exception:
            return f"Bedrock call failed: {exc}"

# Configuration section
with st.sidebar:
    st.header("⚙️ Configuration")
    kb_id = st.text_input("Knowledge Base ID", value=os.getenv("KNOWLEDGE_BASE_ID", "MIRZRYBAU7"))
    model_id = st.text_input("Model ID", value=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0"))
    region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "us-east-1"))
    
    if st.button("Apply settings"):
        os.environ["KNOWLEDGE_BASE_ID"] = kb_id
        os.environ["BEDROCK_MODEL_ID"] = model_id
        os.environ["AWS_REGION"] = region
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("""
    These values come from .env file:
    - KNOWLEDGE_BASE_ID
    - BEDROCK_MODEL_ID
    - AWS_REGION
    """)
    # Debug credentials visible in sidebar (masked secret)
    try:
        creds = load_aws_credentials()
        if creds:
            display_key = creds.get('aws_access_key_id')
            st.write(f"**Loaded credentials (access key):** {display_key}")
            # Test STS with explicit session
            try:
                session = boto3.Session(
                    aws_access_key_id=creds['aws_access_key_id'],
                    aws_secret_access_key=creds['aws_secret_access_key'],
                    region_name=region,
                )
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                st.write(f"**Caller Identity:** {identity.get('Arn')} (Account {identity.get('Account')})")
            except Exception as e:
                st.write(f"**Credential test error:** {e}")
        else:
            st.write("**No credentials found in ~/.aws/credentials**")
    except Exception as e:
        st.write(f"**Credential load error:** {e}")

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
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "MIRZRYBAU7")
    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    
    with st.spinner("Retrieving context and generating answer..."):
        try:
            # Retrieve context from knowledge base and chunk passages
            sources = retrieve_context(prompt, kb_id)
            
            # Generate answer with semantic embedding reranking
            answer = generate_answer(prompt, sources, model_id)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.json(source)
        except Exception as e:
            logger.exception("Error processing query")
            st.error(f"Error: {str(e)}")
