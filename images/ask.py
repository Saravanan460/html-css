import os
import chromadb
from chromadb.utils import embedding_functions
from google.generativeai import GenerativeModel, configure

# ==== CONFIGURATION ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyBxqYdXCm9MPEguZ1oUcpb2jHMcZtnQucw"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_collection(name="company_docs")

# Use the same embedding model (MiniLM)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Function to ask question
def ask_question(query):
    # Embed the question and search similar chunks
    results = collection.query(
        query_texts=[query],
        n_results=2  # top 2 most similar chunks
    )

    docs = results['documents'][0]
    context = "\n\n".join(docs)

    prompt = f"""Answer the following question based on the company policy documents:\n
    Question: {query}\n
    Relevant Context:\n{context}\n
    Answer:"""

    # Use Gemini to answer
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return response.text.strip()

# MAIN
if __name__ == "__main__":
    while True:
        user_q = input("\nAsk a question (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        answer = ask_question(user_q)
        print("\n🧠 Gemini Answer:\n", answer)
