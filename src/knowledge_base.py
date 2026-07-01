class KnowledgeBase:
    def __init__(self):
        self.documents = []

    def add_document(self, document: str) -> None:
        self.documents.append(document)

    def search(self, query: str):
        return [doc for doc in self.documents if query.lower() in doc.lower()]
