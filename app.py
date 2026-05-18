import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Interactive RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── LangChain / HuggingFace imports ──────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    from langchain_community.llms import HuggingFacePipeline           # type: ignore

from transformers import pipeline as hf_pipeline

# Robust RetrievalQA import (works across langchain versions)
try:
    # LangChain 1.x / newer layout
    from langchain.chains.retrieval_qa import RetrievalQA
except Exception:
    try:
        # Older / alternative path
        from langchain.chains import RetrievalQA
    except Exception:
        # Community package fallback
        from langchain_community.chains import RetrievalQA

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "google/flan-t5-large"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 3

# ── Session-state defaults (safe initialisation) ──────────────────────────────
_DEFAULTS = {
    "vectorstore":       None,
    "qa_chain":          None,
    "docs_split":        None,
    "chat_history":      [],
    "retrieved_chunks":  [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders
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


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════
def process_pdfs(uploaded_files):
    """Save uploads to tmp files, load, split, embed → (vectorstore, chunks)."""
    all_docs = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        st.error("No content could be extracted from the uploaded PDF(s).")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks


def build_qa_chain(vs):
    llm = load_llm()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def visualize_chunks(retrieved_docs):
    """Horizontal bar chart showing retrieved chunk relevance ranking."""
    if not retrieved_docs:
        return
    labels = [f"Chunk {i + 1}" for i in range(len(retrieved_docs))]
    scores = list(range(len(retrieved_docs), 0, -1))
    colors = ["#4f8bf9", "#43b8a0", "#f97b4f", "#f9c74f", "#a06cd5"]

    fig, ax = plt.subplots(figsize=(7, max(2, len(retrieved_docs) * 0.9)))
    bars = ax.barh(labels, scores, color=colors[: len(labels)])
    ax.set_xlabel("Relevance Rank (higher = more relevant)")
    ax.set_title("Retrieved Document Chunks")
    ax.invert_yaxis()
    for bar, doc in zip(bars, retrieved_docs):
        src = doc.metadata.get("source", "unknown")
        pg  = doc.metadata.get("page", "?")
        ax.text(
            0.1,
            bar.get_y() + bar.get_height() / 2,
            f"  {os.path.basename(str(src))} — p.{pg}",
            va="center",
            fontsize=8,
            color="white",
        )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — PDF Upload & Settings
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
            st.session_state.vectorstore     = vs
            st.session_state.docs_split      = chunks
            st.session_state.qa_chain        = build_qa_chain(vs)
            st.session_state.chat_history    = []
            st.session_state.retrieved_chunks = []
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
    # ── Chat history ──────────────────────────────────────────────────────────
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
                result      = st.session_state.qa_chain.invoke({"query": query})
                answer      = result["result"]
                source_docs = result.get("source_documents", [])
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.retrieved_chunks = source_docs

    # ── Retrieved chunks visualisation ───────────────────────────────────────
    if st.session_state.retrieved_chunks:
        with st.expander("📊 Retrieved Document Chunks", expanded=True):
            visualize_chunks(st.session_state.retrieved_chunks)
            for i, doc in enumerate(st.session_state.retrieved_chunks):
                src = doc.metadata.get("source", "unknown")
                pg  = doc.metadata.get("page", "?")
                with st.container(border=True):
                    st.markdown(
                        f"**Chunk {i + 1}** — `{os.path.basename(str(src))}` | Page {pg}"
                    )
                    st.text(
                        doc.page_content[:400]
                        + ("…" if len(doc.page_content) > 400 else "")
                    )
