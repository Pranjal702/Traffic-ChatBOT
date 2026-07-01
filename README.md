# Bedrock Chatbot

Local dev repo for the Bedrock knowledge-base chatbot.

This project provides a Streamlit app for a chatbot that answers questions from an Amazon Bedrock knowledge base using Amazon Nova Micro.

## Quick start

1. Create and activate a Python 3.11+ virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) If you want full-quality embeddings, ensure `sentence-transformers` can install native dependencies (PyTorch). If that fails, the project has a deterministic fallback so the app and tests still run.

3. Run tests:

```bash
python -m pytest
```

4. Run the Streamlit app locally:

```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

## Configuration

Set AWS credentials either in `~/.aws/credentials` or via environment variables. Example:

```bash
export AWS_REGION=us-east-1
export KNOWLEDGE_BASE_ID=your-knowledge-base-id
export BEDROCK_MODEL_ID=amazon.nova-micro-v1:0
```

On Windows PowerShell use `setx` or the GUI Credential Manager.

## Notes

- `src/embeddings.py` uses `sentence-transformers` when available; otherwise it provides a deterministic fallback embedding function (useful for CI and constrained environments).
- `src/embedding_reranker.py` implements a cosine-similarity reranker and is preferred by `app.py` when available.
- The app is designed to prefer local embedding-based reranking but will fall back to an in-app scorer when needed.

## CI

A GitHub Actions workflow is included at `.github/workflows/ci.yml` to run tests on push and pull requests.

## Troubleshooting

- If `sentence-transformers` fails to install on Windows due to native DLLs (PyTorch), either install a compatible wheel for your platform or rely on the fallback for development and CI.
