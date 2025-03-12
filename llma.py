import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path = os.getenv("MODEL_PATH", r"D:\PDF-RAG\llama3_hf")

# Step 1: Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
except Exception as e:
    print(e)
    exit(1)

# Create a HuggingFace pipeline
try:
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
except Exception as e:
    print(f"Error creating HuggingFace pipeline: {e}")
    exit(1)
