from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import os

# STEP 1: Read DOCX content
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# STEP 2: Chunk the text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# STEP 3: Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_tensor=False)

# STEP 4: Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection("company_docs")

def store_embeddings(text_chunks, embeddings):
    for i, (text, vector) in enumerate(zip(text_chunks, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[vector.tolist()],
            ids=[f"doc_{i}"]
        )

# STEP 5: Main processing
def process_documents(folder_path):
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            full_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            content = read_docx(full_path)
            chunks = split_text(content)
            all_chunks.extend(chunks)

    print(f"Total Chunks: {len(all_chunks)}")
    embeddings = embed_texts(all_chunks)
    store_embeddings(all_chunks, embeddings)
    print("✅ Embeddings stored in ChromaDB.")

# Run the process
if __name__ == "__main__":
    process_documents("policies")  # folder containing .docx files
