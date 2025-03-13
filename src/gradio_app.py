import gradio as gr
from utils.vector_preprocessing import VectorPreprocessing
from chatbot import Chatbot
from utils.load_config import LoadConfig
import os

# ✅ Load configuration and initialize chatbot
config = LoadConfig()
bot = Chatbot(config)

class GradioChatbot:
    """Manages file uploads and chatbot interactions via Gradio."""

    def __init__(self):
        self.processed_files = set()

    def process_file(self, file_path):
        """Handles file upload and stores embeddings in ChromaDB."""
        if not file_path:
            return "⚠️ No file uploaded. Please select a PDF."

        # Ensure directory exists
        os.makedirs(config.data_directory, exist_ok=True)

        # Save uploaded file
        save_path = os.path.join(config.data_directory, os.path.basename(file_path))
        with open(save_path, "wb") as dest:
            with open(file_path, "rb") as file:
                dest.write(file.read())

        # ✅ Process the file if not already indexed
        if save_path not in self.processed_files:
            vp = VectorPreprocessing(config)
            vp.chunk_and_store(save_path)
            self.processed_files.add(save_path)
            return f"✅ File '{os.path.basename(save_path)}' processed successfully!"
        else:
            return f"⚠️ File '{os.path.basename(save_path)}' is already processed."

    def chat_interface(self, query, history=None):
        """Handles chat interactions in Gradio without history, ensuring proper response formatting."""
        
        prompt = f"""
    Use the following context to answer the user's question. If the context does not contain relevant information, say "I don't know."

    ### CONTEXT:
    {query}

    ### USER QUESTION:

        You are an AI assistant. Answer the following user question based ONLY on the retrieved document content:

        User: {query}
        Bot:
        """

        # ✅ Generate response
        response = bot.chat(prompt)

        # ✅ Extract only the text response if Gemini returns an object
        if hasattr(response, "content"):  # If response is an AIMessage object
            response_text = response.content
        elif isinstance(response, dict) and "content" in response:  # If response is a dictionary
            response_text = response["content"]
        else:  # Fallback to string conversion
            response_text = str(response)

        return response_text  # ✅ Only return the answer, no history
    
# ✅ Initialize Gradio UI
chat_manager = GradioChatbot()

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Chat with Your PDFs (Custom PDF Q&A with Google Gemini)")

    with gr.Row():
        file_upload = gr.File(label="Upload a PDF", type="filepath")  # ✅ Ensure type is 'file'
        process_button = gr.Button("Process PDF")

    process_output = gr.Textbox(label="File Processing Status", interactive=False)
    process_button.click(chat_manager.process_file, inputs=file_upload, outputs=process_output)

    chatbot = gr.ChatInterface(fn=chat_manager.chat_interface)

demo.launch()