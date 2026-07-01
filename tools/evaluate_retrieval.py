"""Small evaluation script to test retrieval + generation locally.

Usage: python tools/evaluate_retrieval.py "What is X?"
"""
import os
import sys
# Ensure project root is on sys.path so `app` can be imported when running from tools/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import retrieve_context, generate_answer


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/evaluate_retrieval.py \"your question\"")
        sys.exit(1)
    query = sys.argv[1]
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "MIRZRYBAU7")
    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

    print(f"Query: {query}")
    sources = retrieve_context(query, kb_id)
    print(f"Retrieved {len(sources)} sources")
    for s in sources:
        print("---")
        print(s.get('id'), s.get('score'))
        print(s.get('source'))
        print(s.get('text')[:500])

    answer = generate_answer(query, sources, model_id)
    print("---\nAnswer:\n", answer)


if __name__ == '__main__':
    main()
