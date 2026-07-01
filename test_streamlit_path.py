import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

print("Environment Check:")
print(f"  KNOWLEDGE_BASE_ID: {os.getenv('KNOWLEDGE_BASE_ID')}")
print(f"  BEDROCK_MODEL_ID: {os.getenv('BEDROCK_MODEL_ID')}")
print(f"  AWS_REGION: {os.getenv('AWS_REGION')}")
print()

# Test import
try:
    from src.chatbot import Chatbot
    print("✓ Chatbot import successful")
except Exception as e:
    print(f"✗ Chatbot import failed: {e}")
    sys.exit(1)

# Test initialization
try:
    chatbot = Chatbot()
    print("✓ Chatbot initialization successful")
except Exception as e:
    print(f"✗ Chatbot initialization failed: {e}")
    sys.exit(1)

# Test retrieval
try:
    print("\nTesting knowledge base retrieval...")
    result = chatbot.ask("What is this knowledge base about?")
    print(f"✓ Query successful!")
    print(f"  Answer: {result.get('answer', '')[:100]}...")
    print(f"  Sources: {len(result.get('sources', []))} results")
except Exception as e:
    print(f"✗ Query failed: {e}")
    sys.exit(1)
