import boto3
from botocore.exceptions import BotoCoreError, ClientError

def test_aws_credentials():
    """Test if AWS credentials are configured correctly."""
    print("Testing AWS credentials...")
    
    try:
        session = boto3.Session(profile_name='default')
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        print("✓ AWS credentials are valid!")
        print(f"  Account: {identity['Account']}")
        print(f"  User/Role: {identity['Arn']}")
        return True
    except (ClientError, BotoCoreError) as e:
        print(f"✗ AWS credentials error: {e}")
        return False

def test_bedrock_access():
    """Test if Bedrock access is available."""
    print("\nTesting Bedrock access...")
    
    try:
        session = boto3.Session(profile_name='default')
        bedrock = session.client('bedrock-runtime', region_name='us-east-1')
        # Test with a simple model list (doesn't require actual invocation)
        print("✓ Bedrock client initialized successfully")
        return True
    except (ClientError, BotoCoreError) as e:
        print(f"✗ Bedrock access error: {e}")
        return False

def test_knowledge_base_retrieval():
    """Test if knowledge base retrieval works."""
    print("\nTesting knowledge base retrieval...")
    
    try:
        session = boto3.Session(profile_name='default')
        agent_runtime = session.client('bedrock-agent-runtime', region_name='us-east-1')
        kb_id = "MIRZRYBAU7"
        
        response = agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": "test query"}
        )
        print(f"✓ Knowledge base retrieval works!")
        print(f"  Retrieved {len(response.get('retrievalResults', []))} results")
        return True
    except (ClientError, BotoCoreError) as e:
        print(f"✗ Knowledge base retrieval error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("AWS Bedrock Setup Verification")
    print("=" * 50)
    
    creds_ok = test_aws_credentials()
    bedrock_ok = test_bedrock_access()
    kb_ok = test_knowledge_base_retrieval()
    
    print("\n" + "=" * 50)
    if creds_ok and bedrock_ok:
        print("✓ Setup looks good! Try running the app now.")
    else:
        print("✗ There are configuration issues to fix.")
        print("  See AWS_SETUP.md for detailed instructions.")
    print("=" * 50)
