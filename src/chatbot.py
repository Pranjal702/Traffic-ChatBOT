from src.bedrock_client import BedrockClient


class Chatbot:
    def __init__(self):
        self.name = "Bedrock Chatbot"
        self.client = BedrockClient()

    def ask(self, prompt: str) -> dict:
        context = self.client.retrieve_context(prompt)
        answer_text = self.client.answer(prompt, context)
        return {"answer": answer_text, "sources": context}
