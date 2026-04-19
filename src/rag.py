import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

HARDCODED_GUIDELINES = """
1. Send SMS/call reminders 48h before appointment for high-risk patients.
2. Offer flexible rescheduling for patients with long lead times (>15 days).
3. Prioritize follow-up calls for patients with no SMS received.
4. Consider same-day confirmation calls for patients with prior no-show history.
5. Engage social workers for patients with multiple chronic conditions.
Source: WHO Health Systems Guidelines, CDC Appointment Adherence Framework.
"""

def retrieve_guidelines(query: str = "care coordination interventions for high-risk patients", top_k: int = 3) -> str:
    """
    Attempts to retrieve guidelines from the local FAISS index.
    Falls back to hardcoded guidelines if the index is missing or an error occurs.
    """
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
    
    if not os.path.exists(index_path):
        return HARDCODED_GUIDELINES
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=top_k)
        retrieved_text = "\n".join([doc.page_content for doc in docs])
        return retrieved_text if retrieved_text else HARDCODED_GUIDELINES
    except Exception as e:
        return HARDCODED_GUIDELINES
