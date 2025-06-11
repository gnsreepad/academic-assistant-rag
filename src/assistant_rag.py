import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

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

docs = load_documents_from_directory("./data/text_book")
chunks = [{"id": f"{d['id']}_chunk{i}", "text": c}
          for d in docs for i, c in enumerate(split_text(d["text"]))]

# Generate embeddings and store in Chroma
for doc in chunks:
    embedding = embedding_model.encode(doc["text"]).tolist()
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])


# Query ChromaDB
def query_documents(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    return [doc for sublist in results["documents"] for doc in sublist]

# Load local LLM (can be Mistral or Phi)
# llm_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Or "microsoft/phi-2" for low-resource
# llm_name = "tiiuae/falcon-rw-1b"

llm_name = "distilgpt2"  # or "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float32  # Force float32 to avoid bfloat16
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model.to(device)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.type == "mps" else -1)


# Generate final answer
def generate_response(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an assistant for question-answering tasks."
        "Use context to answer the question. If you don't know the answer, say that you"
        "Use five sentences maximum.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    output = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# 🔍 Test it!
question = "tell me about the main themes in the book"
chunks = query_documents(question)
response = generate_response(question, chunks)

print(response)