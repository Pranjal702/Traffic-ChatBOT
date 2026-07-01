# AWS Credentials Setup Guide

## Option 1: Using AWS CLI (Recommended)

If you haven't set up AWS credentials yet, use the AWS CLI:

```bash
aws configure
```

When prompted, enter:
- **AWS Access Key ID**: Your access key from AWS IAM
- **AWS Secret Access Key**: Your secret key
- **Default region name**: us-east-1 (or your preferred region)
- **Default output format**: json

This will create credentials in: `C:\Users\Abcom\.aws\credentials`

## Option 2: Using Environment Variables

Set these in PowerShell before running the app:

```powershell
$env:AWS_ACCESS_KEY_ID = "your-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

## Option 3: Using .env File

Add to `bedrock-chatbot/.env`:

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

## Verify Credentials

Test your setup:

```powershell
aws sts get-caller-identity
```

This should return your AWS account info if credentials are valid.

## Bedrock Permissions

Your AWS user/role must have these permissions:
- `bedrock:InvokeModel`
- `bedrock-agent-runtime:Retrieve`

Add this policy to your IAM user if missing:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock-agent-runtime:Retrieve"
      ],
      "Resource": "*"
    }
  ]
}
```

## After Setup

Restart the Streamlit app:

```powershell
cd c:\Users\Abcom\chatBOT\bedrock-chatbot
python -m streamlit run app.py
```
