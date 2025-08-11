import gradio as gr
import tempfile
import os
from src.model.assistant_rag import run_rag_pipeline

# Temporary directory for uploaded files
uploaded_dir = tempfile.TemporaryDirectory()

def save_uploaded_files(files):
    # Clear previous files
    for f in os.listdir(uploaded_dir.name):
        os.remove(os.path.join(uploaded_dir.name, f))

    if not files:
        return "No files uploaded."

    # Save new uploaded files
    for file in files:
        # Get actual file path from NamedString or string
        file_path = file.name if hasattr(file, "name") else file
        with open(file_path, "rb") as f_in:
            content = f_in.read()

        dest_path = os.path.join(uploaded_dir.name, os.path.basename(file_path))
        with open(dest_path, "wb") as f_out:
            f_out.write(content)

    return "Files uploaded successfully! You can now ask questions."

def ask_question(question, chat_history):
    if not question.strip():
        return chat_history, "Please enter a question."

    try:
        # Run the RAG pipeline
        answer = run_rag_pipeline(uploaded_dir.name, question)
    except Exception as e:
        answer = f"Error: {e}"

    chat_history = chat_history or []
    chat_history.append(("User", question))
    chat_history.append(("Assistant", answer))

    return chat_history, ""

with gr.Blocks(title="📚 Document Q&A Assistant") as demo:
    gr.Markdown(
        """
        # Document Q&A Assistant  
        Upload text files and ask questions about their content.  
        Powered by retrieval-augmented generation (RAG) using your documents.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                file_types=[".txt"],
                file_count="multiple",
                label="📂 Upload Text Files"
            )
            upload_button = gr.Button("Upload Files", variant="primary")
            upload_status = gr.Text(value="", interactive=False)

        with gr.Column(scale=2):
            chat = gr.Chatbot(elem_id="chatbot", label="Chat with your documents")
            question_input = gr.Textbox(
                placeholder="Ask a question about the uploaded documents...",
                label="Your Question",
                lines=2,
                max_lines=5
            )
            ask_button = gr.Button("Ask")

    upload_button.click(save_uploaded_files, inputs=file_input, outputs=upload_status)

    ask_button.click(
        ask_question,
        inputs=[question_input, chat],
        outputs=[chat, question_input],
    )

    question_input.submit(
        ask_question,
        inputs=[question_input, chat],
        outputs=[chat, question_input],
    )

    gr.HTML(
        """
        <style>
        #chatbot .chatbot-message-user {
            background-color: #2563eb;
            color: white;
            border-radius: 12px 12px 0 12px;
        }
        #chatbot .chatbot-message-assistant {
            background-color: #f3f4f6;
            color: #111827;
            border-radius: 12px 12px 12px 0;
        }
        </style>
        """
    )

if __name__ == "__main__":
    demo.launch()