import os
from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# === Model & Embeddings Setup ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

def load_documents_from_directory(path):
    docs = []
    for f in os.listdir(path):
        if f.endswith(".txt"):
            with open(os.path.join(path, f), "r", encoding="utf-8") as file:
                docs.append({"id": f, "text": file.read()})
    return docs

def split_text(text, size=1000, overlap=20):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

# Load and embed documents (only run once, or cache appropriately)
docs = load_documents_from_directory("./data/text_book")
chunks = [{"id": f"{d['id']}_chunk{i}", "text": c}
          for d in docs for i, c in enumerate(split_text(d["text"]))]

for doc in chunks:
    embedding = embedding_model.encode(doc["text"]).tolist()
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[embedding])

# === LLM Setup ===
llm_name = "distilgpt2"  # Lightweight model for local use
tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "mps" else -1
)

# === Helper functions ===
def query_documents(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    return [doc for sublist in results["documents"] for doc in sublist]

def generate_response(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use context to answer the question. If you don't know the answer, say that. "
        "Use five sentences maximum.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    output = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# === API Routes ===
@app.route("/askrag", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        context = query_documents(question)
        response = generate_response(question, context)
        return jsonify({
            "question": question,
            "response": response.strip()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"message": "RAG QA API is running"}), 200

# === Run server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)