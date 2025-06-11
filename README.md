# academic-assistant-rag
## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an approach that combines large language models (LLMs) with external knowledge sources. Instead of relying solely on the model's internal knowledge, RAG retrieves relevant documents or data from a knowledge base and uses them to generate more accurate and context-aware responses. This technique is widely used to improve the factual accuracy and relevance of LLM outputs, especially in domains where up-to-date or specialized information is required.

## About This Project

This repository implements a simple academic assistant using the RAG paradigm. The core logic is in `src/app.py`, which orchestrates the retrieval of relevant academic documents and leverages a language model to answer user queries based on the retrieved content.

### Key Features

- **Document Retrieval:** Finds the most relevant academic documents for a given query.
- **Contextual Generation:** Uses the retrieved documents to generate informed and accurate answers.
- **Modular Design:** Easily extendable to support different document sources or language models.


## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/academic-assistant-rag.git
cd academic-assistant-rag
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

- Create two folders in the project root: `chroma_persistent_storage` (for vector database storage) and `data` (for your academic documents).
- Place your academic documents (text files, with .txt extension) in the `data` directory.
- By default, the application is configured to use the `data/text_book` folder as the source for academic documents.  
- If you want to use a different folder, update the path in `src/app.py` where the documents are loaded.  
- Ensure your `.txt` files are placed in the specified directory before running the application.

### 4. Run the Application

```bash
python src/app.py
```

### 6. Customization

- To use a different language model or retrieval backend, modify the relevant sections in `src/app.py`.
- For advanced usage, refer to the code comments and docstrings.

## License

This project is licensed under the MIT License.