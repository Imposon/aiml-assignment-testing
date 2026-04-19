import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

def build_index():
    guideline_path = os.path.join(os.path.dirname(__file__), "guidelines", "care_guidelines.txt")
    if not os.path.exists(guideline_path):
        print(f"Error: Could not find guidelines text at {guideline_path}")
        return
    
    # Load document
    loader = TextLoader(guideline_path)
    documents = loader.load()
    
    # Split text (optional, but good practice. The doc is small though)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the index locally
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    vectorstore.save_local(index_path)
    print(f"Successfully built and saved FAISS index to {index_path}")

if __name__ == "__main__":
    build_index()
