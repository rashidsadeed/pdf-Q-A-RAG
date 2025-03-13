import os
os.environ["CHROMA_DISABLE_ONNX"] = "1"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

class VectorPreprocessing:
    """Handles document loading, chunking, and embedding storage in VectorDB."""
    
    def __init__(self, config):
        self.config = config
        self.embedding = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
        self.vectordb = Chroma(persist_directory=self.config.persist_directory, embedding_function=self.embedding)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, file_path):
        """Load PDF document."""
        if not os.path.exists(file_path):
            print(f"⚠️ Error: PDF file '{file_path}' does not exist!")
            return []

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if docs:
            print(f"✅ Loaded {len(docs)} pages from '{file_path}'.")
        else:
            print(f"⚠️ No text extracted from '{file_path}'. Check if it's a scanned PDF.")
        return docs

    def chunk_and_store(self, file_path):
        """Process and store chunked documents in the vector DB."""
        docs = self.load_documents(file_path)
        if not docs:
            print("⚠️ No documents loaded! Skipping vectorization.")
            return

        chunks = self.text_splitter.split_documents(docs)
        if not chunks:
            print("⚠️ No chunks were created! Ensure the PDF contains extractable text.")
            return
        
        print(f"✅ Storing {len(chunks)} chunks in ChromaDB...")
        self.vectordb.add_documents(chunks)
        print(f"✅ Successfully stored {len(chunks)} chunks in VectorDB.")
