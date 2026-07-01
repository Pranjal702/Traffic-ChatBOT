from src.knowledge_base import KnowledgeBase


def test_kb_add_document():
    kb = KnowledgeBase()
    kb.add_document("Test document")

    assert kb.documents == ["Test document"]
