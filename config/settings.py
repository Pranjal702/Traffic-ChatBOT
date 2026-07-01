import os
from dotenv import load_dotenv

load_dotenv()

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "MIRZRYBAU7")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
