import streamlit as st
import glob
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Embeddings (use langchain-huggingface if available, fall back to community) ──
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline as hf_pipeline
from langchain.chains import RetrievalQA

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-large"
CHUNK_SIZE  = 1000
CHUNK_OVERLAP = 200
TOP_K       = 3

# ── Session-state defaults ────────────────────────────────────────────────────
for key in ["vectorstore", "qa_chain", "docs_split", "chat_history", "retrieved_chunks"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []
        if key == "retrieved_chunks":
            st.session_state[key] = []


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Loading language model (this may take a minute)…")
def load_llm():
    pipe = hf_pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        device_map="auto",
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=pipe)


def process_pdfs(uploaded_files):
    """Load, split, embed uploaded PDFs and return vectorstore + chunks."""
    all_docs = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        all_docs.extend(loader.load())
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks


def build_qa_chain(vectorstore):
    llm = load_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def visualize_chunks(retrieved_docs, all_chunks):
    """Bar chart: relevance rank of retrieved chunks vs. total pool."""
    labels = [f"Chunk {i+1}" for i in range(len(retrieved_docs))]
    scores = list(range(len(retrieved_docs), 0, -1))  # rank proxy

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, scores, color=["#4f8bf9", "#43b8a0", "#f97b4f"][:len(labels)])
    ax.set_xlabel("Relevance Rank (higher = more relevant)")
    ax.set_title("Retrieved Document Chunks")
    ax.invert_yaxis()
    for bar, doc in zip(bars, retrieved_docs):
        src = doc.metadata.get("source", "unknown")
        pg  = doc.metadata.get("page", "?")
        ax.text(0.2, bar.get_y() + bar.get_height()/2,
                f"  {os.path.basename(src)} — p.{pg}",
                va="center", fontsize=8, color="white")
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — PDF Upload
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded and st.button("⚙️ Process Documents"):
        with st.spinner("Processing PDFs and building vector store…"):
            vs, chunks = process_pdfs(uploaded)
            st.session_state.vectorstore  = vs
            st.session_state.docs_split   = chunks
            st.session_state.qa_chain     = build_qa_chain(vs)
            st.session_state.chat_history = []
        st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded)} file(s).")

    st.divider()
    st.markdown("**Models**")
    st.caption(f"🔤 Embeddings: `{EMBED_MODEL}`")
    st.caption(f"🧠 LLM: `{LLM_MODEL}`")
    st.caption(f"📦 Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    st.caption(f"🔍 Top-K retrieval: {TOP_K}")


# ══════════════════════════════════════════════════════════════════════════════
# Main — Chat Interface
# ══════════════════════════════════════════════════════════════════════════════
st.title("🤖 Interactive RAG Chatbot")
st.caption("Ask questions about your uploaded PDF documents.")

if st.session_state.qa_chain is None:
    st.info("👈 Upload PDF(s) from the sidebar and click **Process Documents** to get started.")
else:
    # ── Chat history display ──────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.chat_input("Ask a question about your documents…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.qa_chain.invoke({"query": query})
                answer = result["result"]
                source_docs = result.get("source_documents", [])

            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.retrieved_chunks = source_docs

    # ── Retrieved chunks visualization ───────────────────────────────────────
    if st.session_state.retrieved_chunks:
        with st.expander("📊 Retrieved Document Chunks", expanded=True):
            visualize_chunks(
                st.session_state.retrieved_chunks,
                st.session_state.docs_split,
            )
            for i, doc in enumerate(st.session_state.retrieved_chunks):
                src = doc.metadata.get("source", "unknown")
                pg  = doc.metadata.get("page", "?")
                with st.container(border=True):
                    st.markdown(f"**Chunk {i+1}** — `{os.path.basename(src)}` | Page {pg}")
                    st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))
