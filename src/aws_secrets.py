"""AWS Secrets Manager integration for secure credential retrieval."""
import json
import logging
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


class SecretsManager:
    """Retrieve secrets from AWS Secrets Manager with caching."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = boto3.client("secretsmanager", region_name=region)
        self._cache: Dict[str, Optional[Any]] = {}

    def get_secret(self, secret_arn: str) -> Optional[Dict]:
        """Retrieve a secret by ARN. Results are cached.
        
        Args:
            secret_arn: Full ARN of the secret (e.g., arn:aws:secretsmanager:region:account:secret:name)
            
        Returns:
            Parsed JSON secret as dict, or None if retrieval failed.
        """
        if not secret_arn:
            logger.warning("Secret ARN is empty")
            return None

        if secret_arn in self._cache:
            logger.debug(f"Using cached secret: {secret_arn}")
            return self._cache[secret_arn]

        try:
            logger.info(f"Fetching secret from Secrets Manager: {secret_arn}")
            response = self.client.get_secret_value(SecretId=secret_arn)
            secret_string = response.get("SecretString")
            
            if secret_string:
                try:
                    # Try to parse as JSON
                    parsed = json.loads(secret_string)
                    self._cache[secret_arn] = parsed
                    logger.info(f"Successfully retrieved and cached secret: {secret_arn}")
                    return parsed
                except json.JSONDecodeError:
                    # If not JSON, return as plain string
                    self._cache[secret_arn] = {"value": secret_string}
                    logger.info(f"Secret is plain text, cached as value field: {secret_arn}")
                    return {"value": secret_string}
            
            logger.warning(f"Secret {secret_arn} has no SecretString")
            return None
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "ResourceNotFoundException":
                logger.error(f"Secret not found: {secret_arn}")
            elif error_code == "AccessDeniedException":
                logger.error(f"Access denied to secret: {secret_arn}")
            else:
                logger.error(f"ClientError retrieving secret {secret_arn}: {e}")
        except BotoCoreError as e:
            logger.error(f"BotoCoreError retrieving secret {secret_arn}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret {secret_arn}: {e}")
        
        return None

    def get_knowledge_base_id(self, secret_arn: str) -> Optional[str]:
        """Retrieve Bedrock Knowledge Base ID from Secrets Manager.
        
        Expected format: plain text string OR JSON with 'knowledge_base_id' key.
        
        Args:
            secret_arn: ARN of the secret containing the Knowledge Base ID.
            
        Returns:
            Knowledge Base ID string, or None if not found.
        """
        if not secret_arn:
            logger.error("Knowledge Base Secret ARN not provided")
            return None
            
        secret = self.get_secret(secret_arn)
        if not secret:
            return None
        
        # Try different key names
        for key in ["knowledge_base_id", "KNOWLEDGE_BASE_ID", "kb_id", "value"]:
            if key in secret:
                kb_id = secret[key]
                if kb_id:
                    logger.info(f"Retrieved Knowledge Base ID from key '{key}'")
                    return kb_id
        
        logger.warning(f"Knowledge Base ID not found in secret: {secret_arn}")
        return None
