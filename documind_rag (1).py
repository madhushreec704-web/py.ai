"""
DocuMind AI - PDF Question Answering with RAG + Ollama
Run in Google Colab after installing Ollama and the Python packages.

Required Python packages:
    pip install gradio chromadb pypdf requests

Required Ollama models:
    ollama pull llama3.2:1b
    ollama pull nomic-embed-text

Run:
    python documind_rag.py
"""

import os
import time
import shutil
import subprocess
from pathlib import Path

import requests
import gradio as gr
import chromadb
from pypdf import PdfReader

# -------------------- CONFIGURATION --------------------

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500       # approximate number of words per chunk
CHUNK_OVERLAP = 100
TOP_K = 4
EMBED_BATCH_SIZE = 24

# -------------------- START OLLAMA --------------------

def start_ollama():
    """Start Ollama if it is not already running, then wait for its API."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if response.ok:
            print("Ollama server is already running.")
            return
    except requests.RequestException:
        pass

    ollama_path = shutil.which("ollama")
    if not ollama_path:
        raise RuntimeError(
            "Ollama was not found. In Google Colab, install Ollama first:\n"
            "!curl -fsSL https://ollama.com/install.sh | sh"
        )

    log_file = open("/tmp/documind_ollama.log", "a", buffering=1)
    subprocess.Popen(
        [ollama_path, "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"},
    )

    for _ in range(40):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if response.ok:
                print("Ollama server is ready.")
                return
        except requests.RequestException:
            time.sleep(1)

    log_file.flush()
    raise RuntimeError(
        "Could not start Ollama. Check /tmp/documind_ollama.log for details."
    )


def ensure_model(model_name):
    """Pull a model only if it is not already available locally."""
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
    response.raise_for_status()
    available = {
        item.get("name", "").split(":")[0]
        for item in response.json().get("models", [])
    }
    if model_name.split(":")[0] in available:
        print(f"Model available: {model_name}")
        return

    print(f"Downloading model {model_name}. This may take a few minutes...")
    pull_response = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": model_name, "stream": True},
        stream=True,
        timeout=(30, 1800),
    )
    pull_response.raise_for_status()
    for line in pull_response.iter_lines():
        if line:
            import json
            status = json.loads(line.decode("utf-8"))
            if status.get("status"):
                print(status["status"], end="\r")
    print(f"\nFinished downloading {model_name}.")


# -------------------- RAG STORAGE --------------------

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documind_documents")
document_name = None


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def get_embeddings(texts):
    """Get text embeddings from Ollama's embedding endpoint."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"No embeddings returned by Ollama: {data}")
    return embeddings


def process_pdf(pdf_path):
    """Extract a PDF's text, chunk it, embed it, and store it in ChromaDB."""
    global document_name

    if not pdf_path:
        return "Please upload a PDF first."

    pdf_path = str(pdf_path)
    if not pdf_path.lower().endswith(".pdf"):
        return "Please upload a file with a .pdf extension."

    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {page_number}]\n{page_text}")

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            return (
                "I couldn't extract text from this PDF. "
                "It may be a scanned/image-only PDF; try a text-based PDF."
            )

        chunks = split_text(full_text)
        if not chunks:
            return "No readable text was found in the PDF."

        # Clear old document chunks so questions use the newly uploaded PDF.
        existing = collection.get()
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": Path(pdf_path).name, "chunk": i}
            for i in range(len(chunks))
        ]

        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch_texts = chunks[start:start + EMBED_BATCH_SIZE]
            batch_embeddings = get_embeddings(batch_texts)
            collection.add(
                ids=ids[start:start + len(batch_texts)],
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=metadatas[start:start + len(batch_texts)],
            )

        document_name = Path(pdf_path).name
        return (
            f"Successfully processed **{document_name}**.\n\n"
            f"- Pages with text: {len(pages)}\n"
            f"- Text chunks stored: {len(chunks)}\n"
            "- Your PDF is ready. Open the Chat tab and ask questions!"
        )

    except Exception as exc:
        return f"Error while processing the PDF: {exc}"


# -------------------- QUESTION ANSWERING --------------------

def ask_question(message, history):
    """Retrieve relevant PDF chunks and generate an answer grounded in them."""
    if not message or not message.strip():
        return "Please enter a question."

    if collection.count() == 0:
        return "Please upload and process a PDF in the Upload PDF tab first."

    try:
        query_embedding = get_embeddings([message])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas"],
        )

        retrieved_chunks = results.get("documents", [[]])[0]
        retrieved_metadata = results.get("metadatas", [[]])[0]

        context_parts = []
        for chunk, metadata in zip(retrieved_chunks, retrieved_metadata):
            page_or_chunk = metadata.get("chunk", "unknown")
            context_parts.append(
                f"Source: {metadata.get('source', 'uploaded PDF')}, "
                f"chunk {page_or_chunk}\n{chunk}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are DocuMind AI, an assistant that answers questions
using the uploaded PDF.

Use ONLY the information in the context below to answer the question.
If the answer is not present in the context, clearly say:
"I couldn't find that information in the uploaded document."
Explain concepts in simple, clear language. Do not invent facts.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{message}

ANSWER:"""

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        if not answer:
            return "The language model returned an empty response. Please try again."

        sources = "\n\n**Retrieved sources:** " + ", ".join(
            f"chunk {metadata.get('chunk', '?')}"
            for metadata in retrieved_metadata
        )
        return answer + sources

    except Exception as exc:
        return f"Sorry, something went wrong: {exc}"


# -------------------- GRADIO USER INTERFACE --------------------

def build_app():
    with gr.Blocks(title="DocuMind AI - PDF Assistant") as demo:
        gr.Markdown(
            """
            # 📚 DocuMind AI
            ### Ask questions about your PDF using Retrieval-Augmented Generation (RAG).

            Upload a PDF, process it, and chat with your document.
            """
        )

        with gr.Tab("Upload PDF"):
            pdf_input = gr.File(
                label="Choose a PDF document",
                file_types=[".pdf"],
                type="filepath",
            )
            process_button = gr.Button("Process PDF", variant="primary")
            status_output = gr.Markdown()

            process_button.click(
                fn=process_pdf,
                inputs=pdf_input,
                outputs=status_output,
            )

        with gr.Tab("Chat with PDF"):
            gr.ChatInterface(
                fn=ask_question,
                title="Chat with your document",
                description="Ask a question based on the PDF you processed.",
            )

        gr.Markdown(
            "Built with Python, Ollama, Llama 3.2, Nomic Embeddings, "
            "ChromaDB, PyPDF, and Gradio."
        )

    return demo


if __name__ == "__main__":
    print("Starting DocuMind AI...")
    start_ollama()
    ensure_model(CHAT_MODEL)
    ensure_model(EMBED_MODEL)

    app = build_app()
    # In Colab, share=True creates a public Gradio link for the session.
    app.queue().launch(share=True, debug=True)
