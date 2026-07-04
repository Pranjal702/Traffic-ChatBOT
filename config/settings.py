"""Application configuration - all sensitive data from environment or AWS Secrets Manager.

No hardcoded credentials or defaults. All configuration must be explicitly set.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Bedrock Configuration - Knowledge Base ID from Secrets Manager ARN
KNOWLEDGE_BASE_SECRET_ARN = os.getenv("KNOWLEDGE_BASE_SECRET_ARN")
# Model ID can be set as environment variable
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
