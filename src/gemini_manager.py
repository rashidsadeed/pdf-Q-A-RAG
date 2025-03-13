from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiManager:
    """Handles interactions with Google Gemini API."""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )

    def generate_response(self, prompt):
        """Generate response using Google Gemini."""
        return self.llm.invoke(prompt)
