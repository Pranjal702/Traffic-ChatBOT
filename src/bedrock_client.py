import os
from typing import Any, Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config.settings import AWS_REGION, BEDROCK_MODEL_ID, KNOWLEDGE_BASE_ID


class BedrockClient:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", AWS_REGION)
        self.model_id = os.getenv("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID)
        self.kb_id = os.getenv("KNOWLEDGE_BASE_ID", KNOWLEDGE_BASE_ID).strip()
        
        # Create clients using the default credential chain (reads from ~/.aws/credentials)
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region)
        self.agent_runtime = boto3.client("bedrock-agent-runtime", region_name=self.region)

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        if not self.kb_id:
            return []

        try:
            response = self.agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
            )
            retrieval_results = response.get("retrievalResults", [])
            context: List[Dict[str, Any]] = []
            for item in retrieval_results[:5]:
                content = item.get("content", {})
                text = content.get("text", "")
                source = item.get("location", {})
                context.append({"text": text, "source": source})
            return context
        except (ClientError, BotoCoreError, Exception) as exc:
            return [{"text": f"Knowledge base lookup failed: {exc}", "source": {"sourceType": "Error"}}]

    def answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not self.kb_id:
            return "Set KNOWLEDGE_BASE_ID in your environment to query a Bedrock knowledge base."

        prompt_lines = [
            "You are a helpful assistant. Answer the user's question using the provided context.",
            f"Question: {query}",
            "Context:",
        ]
        if context_chunks:
            prompt_lines.extend(chunk.get("text", "") for chunk in context_chunks)
        else:
            prompt_lines.append("No relevant context was retrieved.")

        prompt = "\n".join(prompt_lines)

        try:
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 400, "temperature": 0.1, "topP": 0.9},
            )
            output = response.get("output", {}).get("message", {}).get("content", [])
            if output:
                return output[0].get("text", "")
            return "No response was returned by Bedrock."
        except (ClientError, BotoCoreError, Exception) as exc:
            return f"Bedrock call failed: {exc}"
