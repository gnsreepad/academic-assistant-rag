# assistant_rag.py
import os
import re
import shutil
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    pipeline
)
import torch
import nltk

# Optional imports for RTF/DOCX parsing
try:
    from striprtf.striprtf import rtf_to_text
except Exception:
    rtf_to_text = None

try:
    import docx
except Exception:
    docx = None

# Ensure sentence tokenizer is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab")

# ---------------------------
# Embedding Model
# ---------------------------
def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    """
    Returns a Chroma-compatible SentenceTransformer embedding function.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


# ---------------------------
# ChromaDB Collection helper
# ---------------------------
def get_chroma_collection(collection_name, storage_path, embedding_fn):
    chroma_client = chromadb.PersistentClient(path=storage_path)
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )


# ---------------------------
# Text extraction for various file types
# ---------------------------
def extract_text_from_file(full_path):
    _, ext = os.path.splitext(full_path)
    ext = ext.lower()

    if ext == ".txt":
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".docx":
        if docx is None:
            raise ImportError("Install python-docx (pip install python-docx) to parse .docx files.")
        doc = docx.Document(full_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    if ext == ".rtf":
        if rtf_to_text is None:
            raise ImportError("Install striprtf (pip install striprtf) to parse .rtf files.")
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            rtf_raw = f.read()
        return rtf_to_text(rtf_raw)

    raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------
# Cleaning & chunking
# ---------------------------
def clean_text(text):
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable chars
    text = "".join(c for c in text if c.isprintable())
    return text.strip()


def split_text(text, max_chunk_size=500, overlap_sentences=2):
    """
    Split text into chunks by sentences.
    - max_chunk_size: approx char limit per chunk (tune to model)
    - overlap_sentences: how many sentences to overlap between chunks (keeps context)
    """
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        s_len = len(sent)
        # if sentence fits into current chunk, append
        if current_len + s_len + 1 <= max_chunk_size:
            current_chunk.append(sent)
            current_len += s_len + 1
            continue

        # otherwise flush current chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        # prepare next chunk with overlap sentences
        if overlap_sentences > 0:
            overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else current_chunk
            current_chunk = list(overlap)
            current_len = sum(len(s) + 1 for s in current_chunk)
        else:
            current_chunk = []
            current_len = 0

        # if this single sentence alone is too long, truncate it
        if s_len > max_chunk_size:
            chunks.append(sent[:max_chunk_size].strip())
            current_chunk = []
            current_len = 0
        else:
            current_chunk.append(sent)
            current_len += s_len + 1

    # flush last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # filter out empty/very short chunks
    chunks = [c for c in chunks if len(c) > 20]
    return chunks


# ---------------------------
# Load documents from directory (supports .txt, .docx, .rtf)
# ---------------------------
def load_documents_from_directory(path):
    docs = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    for fname in sorted(os.listdir(path)):
        full_path = os.path.join(path, fname)
        if not os.path.isfile(full_path):
            continue
        if not fname.lower().endswith((".txt", ".docx", ".rtf")):
            continue
        try:
            raw = extract_text_from_file(full_path)
            cleaned = clean_text(raw)
            if cleaned:
                docs.append({"id": fname, "text": cleaned})
        except Exception as e:
            print(f"[WARNING] Skipping {fname}: {e}")
    return docs


# ---------------------------
# Store documents in Chroma (assumes fresh DB)
# ---------------------------
def store_documents_in_chroma(docs, collection, max_chunk_size=500, overlap_sentences=2):
    for d in docs:
        chunks = split_text(d["text"], max_chunk_size=max_chunk_size, overlap_sentences=overlap_sentences)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{d['id']}_chunk{i}"
            collection.add(ids=[chunk_id], documents=[chunk])


# ---------------------------
# Query relevant chunks
# ---------------------------
def query_documents(collection, question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    # results["documents"] is list-of-lists (one list per query)
    return [doc for sub in results.get("documents", []) for doc in sub]


# ---------------------------
# Model loader: choose Seq2Seq vs Causal automatically
# ---------------------------
def load_model_and_tokenizer(model_name_or_path, device=None, torch_dtype=torch.float32):
    print(f"[INFO] Loading model: {model_name_or_path}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        task = "text-generation"

    # move model to device (pipeline will use device index)
    model.to(device)

    pipeline_device = -1 if device in ("cpu", "mps") else 0
    generator = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device,
        do_sample=True,
        temperature=0.0,   # lower temp for more factual answers; adjust if you want more creativity
        max_new_tokens=150,
    )
    return generator


# ---------------------------
# Build prompt and generate answer (debug prints)
# ---------------------------
def generate_response(generator, question, context_chunks, max_prompt_chars=2000):
    # assemble context but cap prompt length to avoid model input overflow
    context = "\n\n".join(context_chunks)
    # truncate context to last characters that keep prompt under limit
    if len(context) > max_prompt_chars:
        context = context[-max_prompt_chars:]

    prompt = f"""
        You are a helpful assistant for question-answering tasks.
        Use ONLY the given context to answer the question.
        If the context does not contain the answer, say "I don't know."
        Write an informative and detailed answer of at least four sentences.
        Elaborate fully and include relevant examples from the context.

        Context:
        {context}

        Question:
        {question}
        Answer:
        """

        
                

    print("\n[DEBUG] Prompt (first 1200 chars):\n", prompt[:1200], "\n--- end prompt ---\n")
    # out = generator(prompt, max_new_tokens=200, do_sample=False)
    out = generator(
            prompt,
            max_new_tokens=200,
            min_new_tokens=80,  # forces more length
            do_sample=False
        )
    #out = generator(prompt, max_length=256, temperature=0.7, do_sample=True)
    # generator returns a list of outputs; extract text
    text = out[0].get("generated_text") if isinstance(out[0], dict) else out[0][0]
    return text


# ---------------------------
# Main RAG pipeline
# ---------------------------
def run_rag_pipeline(text_dir_path, question,
                     storage_path="chroma_persistent_storage",
                     embedding_model_name="all-MiniLM-L6-v2",
                     model_name_env=None,
                     max_chunk_size=500,
                     overlap_sentences=2,
                     n_results=2):
    # remove old DB so each run is fresh (helpful during debugging)
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
        print("[DEBUG] Cleared old vector DB storage.")

    # create collection with embedding function
    embedding_fn = get_embedding_function(model_name=embedding_model_name)
    collection = get_chroma_collection(
        collection_name="document_qa_collection",
        storage_path=storage_path,
        embedding_fn=embedding_fn
    )

    # load & clean documents
    docs = load_documents_from_directory(text_dir_path)
    if not docs:
        return "No documents found in the directory."

    # store chunks
    store_documents_in_chroma(docs, collection, max_chunk_size=max_chunk_size, overlap_sentences=overlap_sentences)

    # retrieve
    context_chunks = query_documents(collection, question, n_results=n_results)

    print("\n[DEBUG] Retrieved context chunks:")
    for i, c in enumerate(context_chunks, start=1):
        print(f"--- Chunk {i} (len {len(c)} chars) ---")
        print(c[:500] + ("\n..." if len(c) > 500 else ""))
        print()

    # choose model
    model_name = model_name_env or os.getenv("MODEL_NAME") or "google/flan-t5-base"
    generator = load_model_and_tokenizer(model_name)

    answer = generate_response(generator, question, context_chunks)
    return answer


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    text_dir_path = "./data/text_book"
    question = "What is the text talking about?"

    ans = run_rag_pipeline(text_dir_path, question)
    print("\n--- ANSWER ---\n")
    print(ans)