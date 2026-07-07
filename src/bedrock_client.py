"""Bedrock integration for document retrieval and response generation.

All configuration (Knowledge Base ID, Model ID) is fetched from AWS Secrets Manager.
"""
import logging
import re
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.aws_secrets import SecretsManager

logger = logging.getLogger(__name__)

# Unicode range for Devanagari script (covers Hindi)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common greetings in both languages — handled without hitting the model,
# so "hi" never accidentally gets treated as a request for KB content.
_GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "hola",
    "namaste", "namaskar", "namaskaar", "good morning",
    "good afternoon", "good evening",
}


def detect_language(query: str) -> str:
    """Detect whether a query is Hindi or English.

    Strategy:
    1. If the text contains any Devanagari characters, it's Hindi —
       this is 100% reliable and doesn't depend on any library.
    2. Otherwise, treat it as English. We deliberately do NOT run a
       statistical language detector (e.g. langdetect) on short strings
       like "hi" or "fine?" — those detectors are unreliable below
       ~20 characters and can misfire, which is what was causing the
       original bug. Romanized Hindi (Hinglish) is treated as English
       by design, since the KB content itself is expected to be in
       English or Devanagari Hindi, not transliterated Hindi.
    """
    if not query:
        return "English"
    if _DEVANAGARI_RE.search(query):
        return "Hindi"
    return "English"


def is_greeting(query: str) -> bool:
    """Check if the message is just a greeting with no real question."""
    normalized = query.strip().lower().rstrip("!.?")
    return normalized in _GREETINGS


class BedrockClient:
    """Interact with AWS Bedrock for retrieval and generation.

    Knowledge Base ID is fetched from AWS Secrets Manager.
    """

    def __init__(
        self,
        kb_id_secret_arn: Optional[str] = None,
        model_id: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """Initialize Bedrock client.

        Args:
            kb_id_secret_arn: AWS Secrets Manager ARN containing Knowledge Base ID.
            model_id: Bedrock Model ID (from environment, not hardcoded).
            region: AWS region.

        Raises:
            ValueError: If Knowledge Base ID cannot be retrieved from Secrets Manager.
        """
        self.region = region
        self.kb_id = None
        self.model_id = model_id or "amazon.nova-micro-v1:0"

        # Retrieve Knowledge Base ID from Secrets Manager
        if kb_id_secret_arn:
            secrets_mgr = SecretsManager(region=region)
            self.kb_id = secrets_mgr.get_knowledge_base_id(kb_id_secret_arn)

            if not self.kb_id:
                raise ValueError(
                    f"Could not retrieve Knowledge Base ID from Secrets Manager: {kb_id_secret_arn}"
                )
            logger.info("Knowledge Base ID retrieved from Secrets Manager")
        else:
            raise ValueError("KNOWLEDGE_BASE_SECRET_ARN environment variable must be set")

        # Initialize Bedrock clients
        try:
            self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
            self.agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region)
            logger.info(f"Initialized Bedrock clients for region {region}")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to initialize Bedrock clients: {e}")
            raise

    def retrieve_context(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Retrieve context from Bedrock Knowledge Base.

        Args:
            query: User query.
            top_k: Maximum number of results.

        Returns:
            List of context chunks with text, source, and similarity scores.
        """
        if not self.kb_id:
            logger.warning("Knowledge Base ID not set; cannot retrieve")
            return []

        try:
            logger.debug(f"Retrieving from Bedrock KB: {self.kb_id[:20]}...")
            response = self.agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
            )
            retrieval_results = response.get("retrievalResults", [])
            chunks = []

            for doc_idx, item in enumerate(retrieval_results[:top_k], start=1):
                content = item.get("content", {})
                text = content.get("text", "")
                source = item.get("location", {})
                score = item.get("score") or item.get("similarity") or 0.0
                doc_id = f"SRC_{doc_idx}"

                chunks.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "source": {"docId": doc_id, "location": source},
                        "score": float(score),
                    }
                )

            logger.info(f"Retrieved {len(chunks)} chunks from Bedrock KB")
            return chunks

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Bedrock retrieval failed: {e}")
            return []

    def generate_answer(
        self, query: str, context_chunks: List[Dict[str, Any]], max_tokens: int = 350
    ) -> str:
        """Generate an answer using Bedrock with provided context.

        Args:
            query: User query.
            context_chunks: Retrieved context.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated answer text.
        """
        detected_lang = detect_language(query)

        # Handle pure greetings without ever touching the model or KB —
        # this was one source of odd behavior ("hi" pulling in Hindi
        # traffic-rule framing from the system prompt).
        if is_greeting(query):
            return "Hello! How can I help you with traffic rules today?" \
                if detected_lang == "English" \
                else "नमस्ते! मैं ट्रैफिक नियमों से जुड़े आपके सवालों में कैसे मदद कर सकता हूं?"

        if not context_chunks:
            return (
                "I don't know — no evidence in knowledge base."
                if detected_lang == "English"
                else "मुझे जानकारी नहीं है — नॉलेज बेस में कोई प्रमाण नहीं मिला।"
            )

        passage_texts = []
        for chunk in context_chunks:
            source_label = chunk.get("source", {}).get("location", {}).get("uri", "unknown")
            passage_texts.append(
                f"[{chunk['id']}] Source: {source_label}\n{chunk.get('text', '')[:1500]}"
            )

        prompt = f"""You are an Indian traffic rules assistant.

LANGUAGE RULE (highest priority — follow this exactly):
Detected question language: {detected_lang}
You MUST write your entire answer in {detected_lang}, regardless of what language the passages below are written in. Passages may be in Hindi even when you must answer in English, and vice versa — the passage language never determines your answer language.

GROUNDING RULE:
- Use ONLY the passages provided below. Do not use outside knowledge.
- Do not invent rule numbers, section names, or details that are not explicitly present in the passages.
- If the passages don't contain the answer, say so plainly in {detected_lang} instead of guessing.

STYLE RULE:
- Keep the answer concise, factual, and professional.
- Do not restate the question or add filler before the answer.

QUESTION:
{query}

PASSAGES:
{chr(10).join(passage_texts)}

Answer (in {detected_lang} only):"""

        try:
            logger.debug(f"Generating answer with model: {self.model_id}")
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0, "topP": 0.9},
            )
            output = response.get("output", {}).get("message", {}).get("content", [])
            if output:
                return output[0].get("text", "")
            return "No response was returned by Bedrock."
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Bedrock generation failed: {e}")
            return f"Error generating response: {e}"
