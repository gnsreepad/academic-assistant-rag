import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    pipeline
)
import torch


# ---------------------------
# Embedding Model
# ---------------------------
def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


# ---------------------------
# ChromaDB Collection
# ---------------------------
def get_chroma_collection(collection_name, storage_path, embedding_fn):
    chroma_client = chromadb.PersistentClient(path=storage_path)
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )


# ---------------------------
# Load .txt Documents
# ---------------------------
def load_documents_from_directory(path):
    docs = []
    for f in os.listdir(path):
        if f.endswith(".txt"):
            with open(os.path.join(path, f), "r", encoding="utf-8") as file:
                docs.append({"id": f, "text": file.read()})
    return docs


# ---------------------------
# Split Text into Chunks
# ---------------------------
def split_text(text, size=1000, overlap=20):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]


# ---------------------------
# Store Docs in Chroma
# ---------------------------
def store_documents_in_chroma(docs, collection):
    for d in docs:
        chunks = split_text(d["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{d['id']}_chunk{i}"
            # Check if chunk already exists to avoid duplicates
            existing = collection.get(ids=[chunk_id])
            if not existing["ids"]:
                collection.add(ids=[chunk_id], documents=[chunk])


# ---------------------------
# Query Relevant Docs
# ---------------------------
def query_documents(collection, question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    # Flatten list of chunks
    return [doc for sublist in results["documents"] for doc in sublist]


# ---------------------------
# Load model and tokenizer with auto-detection
# ---------------------------
def load_model_and_tokenizer(model_name_or_path, device=None, torch_dtype=torch.float32):
    print(f"Loading model: {model_name_or_path}")

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if config.is_encoder_decoder:
        print("Detected Encoder-Decoder model (T5/BART style).")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype
        )
    else:
        print("Detected Decoder-Only model (GPT/LLaMA style).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype
        )

    # Move model to device
    model.to(device)

    # Set pipeline device param:
    pipeline_device = -1 if device in ["cpu", "mps"] else 0

    # Create pipeline according to model type
    if config.is_encoder_decoder:
        task = "text2text-generation"
    else:
        task = "text-generation"

    generator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=150,
    )

    return generator


# ---------------------------
# Generate Answer from retrieved context
# ---------------------------
def generate_response(generator, question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant for question-answering tasks.\n"
        "Use the given context to answer the question. "
        "If you do not know the answer, say that you don't know. "
        "Limit your answer to five sentences.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    )
    output = generator(prompt)
    return output[0]["generated_text"]


# ---------------------------
# Main RAG Pipeline
# ---------------------------
def run_rag_pipeline(text_dir_path, question):
    embedding_fn = get_embedding_function()
    collection = get_chroma_collection(
        collection_name="document_qa_collection",
        storage_path="chroma_persistent_storage",
        embedding_fn=embedding_fn
    )

    docs = load_documents_from_directory(text_dir_path)
    store_documents_in_chroma(docs, collection)

    context_chunks = query_documents(collection, question, n_results=3)  # tweak n_results


    model_name = os.getenv("MODEL_NAME", "google/flan-t5-base")

    generator = load_model_and_tokenizer(model_name)
    answer = generate_response(generator, question, context_chunks)
    return answer


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
 
    text_dir_path = "./data/text_book"
    question = "Tell me about the main themes in the book."

    answer = run_rag_pipeline(text_dir_path, question)
    print("\n--- ANSWER ---\n")
    print(answer)