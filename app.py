import gradio as gr
import tempfile
import os
from src.model.assistant_rag import run_rag_pipeline

# Temporary directory for uploaded files
uploaded_dir = tempfile.TemporaryDirectory()

def save_uploaded_files(files):
    # Clear previous files in the upload directory
    for f in os.listdir(uploaded_dir.name):
        try:
            os.remove(os.path.join(uploaded_dir.name, f))
        except Exception:
            pass

    if not files:
        return "No files uploaded."

    for file in files:
        file_path = getattr(file, "name", None) or file
        try:
            with open(file_path, "rb") as f_in:
                content = f_in.read()
            dest_path = os.path.join(uploaded_dir.name, os.path.basename(file_path))
            with open(dest_path, "wb") as f_out:
                f_out.write(content)
        except Exception as e:
            return f"Error saving file {file_path}: {e}"

    return "Files uploaded successfully! You can now ask questions."

def ask_question(question, chat_history):
    if not question.strip():
        return chat_history, "Please enter a question."

    chat_history = chat_history or []
    chat_history.append(("User", question))

    try:
        answer = run_rag_pipeline(uploaded_dir.name, question)
    except Exception as e:
        answer = f"Error: {e}"

    chat_history.append(("Assistant", answer))
    return chat_history, ""

with gr.Blocks(title="Document Q&A Assistant") as demo:
    gr.Markdown(
        """
        # Document Q&A Assistant  
        Upload **.txt** files and ask questions about their content using RAG.  
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
                placeholder="Type your question here...",
                label="Your Question",
                lines=2,
                max_lines=5
            )
            ask_button = gr.Button("Ask", variant="secondary")

    upload_button.click(
        save_uploaded_files, 
        inputs=file_input, 
        outputs=upload_status
    )

    ask_button.click(
        ask_question,
        inputs=[question_input, chat],
        outputs=[chat, question_input]
    )

    question_input.submit(
        ask_question,
        inputs=[question_input, chat],
        outputs=[chat, question_input]
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