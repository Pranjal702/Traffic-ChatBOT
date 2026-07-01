from src.chatbot import Chatbot


class DummyClient:
    def retrieve_context(self, query):
        return [{"text": "Relevant passage", "source": {"sourceType": "S3"}}]

    def answer(self, query, context_chunks):
        return "Answered from knowledge base"


def test_chatbot_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr("src.chatbot.BedrockClient", lambda: DummyClient())

    chatbot = Chatbot()
    result = chatbot.ask("What is this about?")

    assert result["answer"] == "Answered from knowledge base"
    assert result["sources"] == [{"text": "Relevant passage", "source": {"sourceType": "S3"}}]
