import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSeq2SeqLM,
    pipeline
)
import torch
import nltk

# Download punkt tokenizer once
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab')


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
    
    # Delete existing collection if it exists
    if collection_name in chroma_client.list_collections():
        chroma_client.delete_collection(name=collection_name)
    
    # Then create a new empty collection
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
# Split Text into Overlapping Chunks (robust chunking)
# ---------------------------
def split_text(text, max_chunk_size=900, overlap_chars=150):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            # Keep last `overlap_chars` from current chunk as overlap for next chunk
            current_chunk = current_chunk[-overlap_chars:] + " " + sent
        else:
            current_chunk += " " + sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ---------------------------
# Store Docs in ChromaDB (with chunking)
# ---------------------------
def store_documents_in_chroma(docs, collection, max_chunk_size=900, overlap_chars=150):
    for d in docs:
        chunks = split_text(d["text"], max_chunk_size=max_chunk_size, overlap_chars=overlap_chars)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{d['id']}_chunk{i}"
            # Check if chunk already exists to avoid duplicates
            existing = collection.get(ids=[chunk_id])
            if not existing["ids"]:
                collection.add(ids=[chunk_id], documents=[chunk])


# ---------------------------
# Query Relevant Docs & Deduplicate
# ---------------------------
def query_documents(collection, question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = [doc for sublist in results["documents"] for doc in sublist]
    # Deduplicate chunks by content
    seen = set()
    unique_docs = []
    for d in docs:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)
    return unique_docs


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
        raise NotImplementedError("This pipeline supports only encoder-decoder models for now.")

    model.to(device)

    pipeline_device = -1 if device in ["cpu", "mps"] else 0

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device,
        do_sample=True,
        temperature=0.8,
        max_new_tokens=450,
        top_p=0.95,
        repetition_penalty=1.3
    )

    return generator


# ---------------------------
# Generate Answer from retrieved context
# ---------------------------
def generate_response(generator, question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
            You are a helpful assistant for question-answering tasks.
            Use ONLY the given context to answer the question.
            If the context does not contain the answer, say "I don't know."
            Explain the historical context, causes, and consequences described in the text, using complete sentences and smooth transitions.
            Write a clear, detailed answer.

            Context:
            {context}

            Question:
            {question}
            Answer:
            """
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

    context_chunks = query_documents(collection, question, n_results=5)

    model_name = os.getenv("MODEL_NAME", "google/flan-t5-large")  # Use large model by default

    generator = load_model_and_tokenizer(model_name)
    answer = generate_response(generator, question, context_chunks)
    return answer


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    text_dir_path = "./data/text_book"
    question = "Tell me a 100 character summary?"

    answer = run_rag_pipeline(text_dir_path, question)
    print("\n--- ANSWER ---\n")
    print(answer)
