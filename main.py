from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # Updated import
from langchain.chains import RetrievalQA
from llma import llm


# read the PDF file
pdf_path = r"D:\PDF-RAG\test.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()


# chunk text into smaller parts
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

#generate embeddings and store in vector DB
embedding_model  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_dir = "./vector_db"
vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory= persist_dir)
vector_store.persist()


#similarity search implementation
retriever = vector_store.as_retriever()



#formatting queries for the LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

#query the LLM
query = "what is the docuement about?"
response = qa_chain.run(query)

print(response)

print("-----------------")
print("system operational")
print("-----------------")