from utils.load_config import LoadConfig
from utils.vector_preprocessing import VectorPreprocessing

config = LoadConfig()
vp = VectorPreprocessing(config)

pdf_path = "data\docs\An Introduction to Game Theory.pdf"  # Update with your PDF path
print(f"🔍 Re-indexing '{pdf_path}' in ChromaDB...")
vp.chunk_and_store(pdf_path)

print("\n✅ Re-indexing complete. Run `test_vectordb.py` to verify.")
