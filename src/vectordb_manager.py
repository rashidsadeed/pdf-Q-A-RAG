from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorDBManager:
    """Handles retrieval from ChromaDB."""
    
    def __init__(self, config):
        self.config = config
        self.embedding = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
        self.vectordb = Chroma(persist_directory=config.persist_directory, embedding_function=self.embedding)

    def retrieve_relevant_docs(self, query):
        """Retrieve similar document chunks."""
        return self.vectordb.similarity_search(query, k=self.config.k)
