# 🤖 Interactive RAG Chatbot

A Streamlit application that lets you chat with your PDF documents using:
- **FAISS** vector store for fast similarity retrieval
- **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- **Google Flan-T5-Large** as the local language model
- **LangChain** `RetrievalQA` chain to tie it all together
- **Chunk visualization** showing which document parts were retrieved for each answer

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Zekeriya-Ui/interactive-rag-chatbot.git
cd interactive-rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

## 🖥️ Usage

1. Open the app in your browser (default: `http://localhost:8501`).
2. In the **sidebar**, upload one or more PDF files.
3. Click **⚙️ Process Documents** — embeddings are built and stored in FAISS.
4. Type your question in the chat input at the bottom.
5. The app displays the answer **and** a bar chart of the retrieved document chunks with source/page metadata.

## 📁 Project Structure

```
interactive-rag-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## ⚙️ Configuration

Edit the constants at the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `google/flan-t5-large` | Text generation model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `3` | Number of chunks to retrieve |

## 📦 Key Dependencies

- [Streamlit](https://streamlit.io/) — web UI framework
- [LangChain](https://python.langchain.com/) — RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — local LLM & embeddings
- [PyPDF](https://pypdf.readthedocs.io/) — PDF loading

## 🔑 Optional: HuggingFace Token

Set `HF_TOKEN` as an environment variable or in `.streamlit/secrets.toml` to avoid rate-limit warnings when downloading models:

```toml
# .streamlit/secrets.toml
HF_TOKEN = "hf_your_token_here"
```

## 📄 License

MIT
