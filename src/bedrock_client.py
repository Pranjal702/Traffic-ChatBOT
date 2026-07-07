"""Bedrock integration for document retrieval and response generation.

All configuration (Knowledge Base ID, Model ID) is fetched from AWS Secrets Manager.
"""
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.aws_secrets import SecretsManager

logger = logging.getLogger(__name__)


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

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], max_tokens: int = 350) -> str:
        """Generate an answer using Bedrock with provided context.
        
        Args:
            query: User query.
            context_chunks: Retrieved context.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Generated answer text.
        """
        if not context_chunks:
            return "I don't know — no evidence in knowledge base."

        passage_texts = []
        for chunk in context_chunks:
            source_label = chunk.get("source", {}).get("location", {}).get("uri", "unknown")
            passage_texts.append(
                f"[{chunk['id']}] Source: {source_label}\n{chunk.get('text', '')[:1500]}"
            )

        prompt = (
            "You are a indian traffic chatbot assistant. Greet the User and answeer the questions about the traffic rules in a professional manner, and do not use extra words just wish back whem User is wishing. Answer in the same language as the question using ONLY the passages below. "
            "If the context mentions chapter titles or section names related to the question, use them to infer the topic. Parse english and hindi words carefully, and only reply in USER Language, english(primary) hindi (secondary)."
            "Keep the answer concise, factual, and based only on the provided context.\n\n"
            f"QUESTION:\n{query}\n\nPASSAGES:\n" + "\n\n".join(passage_texts) + "\n\nAnswer:"
        )

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
