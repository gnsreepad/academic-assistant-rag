import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# Load and chunk documents
def load_documents_from_directory(path):
    docs = []
    for f in os.listdir(path):
        if f.endswith(".txt"):
            with open(os.path.join(path, f), "r", encoding="utf-8") as file:
                docs.append({"id": f, "text": file.read()})
    return docs

def split_text(text, size=1000, overlap=20):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

docs = load_documents_from_directory("./news_articles")
chunks = [{"id": f"{d['id']}_chunk{i}", "text": c}
          for d in docs for i, c in enumerate(split_text(d["text"]))]

# Generate embeddings and store in Chroma
for doc in chunks:
    embedding = embedding_model.encode(doc["text"]).tolist()
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])

