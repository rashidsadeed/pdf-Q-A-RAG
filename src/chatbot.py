from gemini_manager import GeminiManager
from vectordb_manager import VectorDBManager

class Chatbot:
    """Manages the chatbot workflow including retrieval and generation."""
    
    def __init__(self, config):
        self.llm_manager = GeminiManager(config)
        self.vector_db = VectorDBManager(config)

    def format_prompt(self, retrieved_text, query):
        """Format prompt in a structured way to guide Gemini."""
        return f"""
Use the following context to answer the user's question. If the context does not contain relevant information, say "I don't know."

### CONTEXT:
{retrieved_text if retrieved_text else "No relevant information available."}

### USER QUESTION:
{query}

### RESPONSE:
"""

    def chat(self, query):
        """Retrieve documents, format prompt, and generate a response."""
        retrieved_docs = self.vector_db.retrieve_relevant_docs(query)
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Debugging: Print Retrieved Content
        print("\n🔍 Retrieved Text from VectorDB:\n", retrieved_text)

        if not retrieved_text.strip():
            print("⚠️ No relevant content retrieved! Ensure the PDF is indexed in VectorDB.")
            return "⚠️ No relevant content found in the database. Please try a different query."

        prompt = self.format_prompt(retrieved_text, query)

        # Debugging: Print the Final Prompt Sent to Gemini
        print("\n🔍 Final Prompt Sent to Gemini:\n", prompt)

        response = self.llm_manager.generate_response(prompt)
        return response
