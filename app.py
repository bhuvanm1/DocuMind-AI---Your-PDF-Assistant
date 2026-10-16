"""DocuMind AI — Your PDF Research Assistant."""
from pathlib import Path
import streamlit as st

from rag.llm import OllamaError, answer, status
from rag.pdf_processor import PDFProcessingError
from rag.pipeline import ask, index_pdf
from rag.vector_store import VectorStore

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(145deg,#f6f8fc 0%,#eef3fb 100%);}
.block-container {max-width: 1120px; padding-top: 2rem;}
section[data-testid="stMain"] .stMarkdown p,
section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {color:#24334b !important;}
[data-testid="stSidebar"] {background: #101b35; color: #f6f8ff;}
[data-testid="stSidebar"] * {color: #f6f8ff;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {background:#192744; border:1px dashed #647694;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] .stButton button {background:#243553 !important; color:#f6f8ff !important; border:1px solid #52627f !important;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background:#192744 !important; color:#f6f8ff !important; border-color:#52627f !important;}
[data-testid="stSidebar"] [data-baseweb="tag"] {background:#355a91 !important; color:#ffffff !important;}
[data-testid="stSidebar"] [data-baseweb="tag"] svg {fill:#ffffff !important;}
.hero {padding:1.6rem 1.8rem;border-radius:22px;background:linear-gradient(120deg,#142b57,#3d70c9);color:white;margin-bottom:1rem;box-shadow:0 12px 34px #263f6825;}
.hero h1 {color:white;margin:0;font-size:2rem;} .hero p {color:#e3ecff;margin:.35rem 0 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(str(Path(__file__).parent / ".chroma"))

store = get_store()
if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Welcome to DocuMind AI. Upload one or more research PDFs in the sidebar, select the documents to search, and ask a question. I’ll show the retrieved passages and page citations so you can check each answer."}]
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

with st.sidebar:
    st.markdown("## 📚 DocuMind AI")
    st.caption("Your private, local PDF research assistant")
    uploaded = st.file_uploader("Add PDF", type=["pdf"], help="PDF only · up to 20 MB")
    if uploaded is not None:
        if uploaded.size > 20 * 1024 * 1024:
            st.error("This file exceeds the 20 MB upload limit.")
        else:
            try:
                with st.spinner("Extracting text and building the local index…"):
                    doc_id, created = index_pdf(store, uploaded.getvalue(), uploaded.name)
                st.session_state.selected_docs = list(dict.fromkeys([*st.session_state.selected_docs, doc_id]))
                st.success("PDF indexed." if created else "This PDF is already in your local library.")
            except PDFProcessingError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not index the PDF: {exc}")
    documents = store.list_documents()
    st.markdown("### Your documents")
    if documents:
        options = {f"{d['filename']} · {d['document_id'][:8]}": d["document_id"] for d in documents}
        selected = st.multiselect("Search within", list(options), default=[k for k,v in options.items() if v in st.session_state.selected_docs], label_visibility="collapsed")
        st.session_state.selected_docs = [options[k] for k in selected]
        for label, doc_id in options.items():
            c1, c2 = st.columns([4, 1])
            c1.caption(label)
            if c2.button("✕", key=f"delete_{doc_id}", help=f"Delete {label}"):
                store.delete_document(doc_id)
                st.session_state.selected_docs = [x for x in st.session_state.selected_docs if x != doc_id]
                st.rerun()
    else:
        st.caption("No PDFs indexed yet.")
    retrieval_count = st.slider("Retrieved passages", min_value=1, max_value=10, value=5)
    ollama_ok, model_ok, detail = status()
    if not ollama_ok:
        st.warning("Ollama is not reachable. Install and start Ollama, then reopen this app.")
    elif not model_ok:
        st.warning("Model qwen2.5:3b is missing. In a terminal run: ollama pull qwen2.5:3b")
    else:
        st.success("Ollama ready · qwen2.5:3b")

st.markdown('<div class="hero"><h1>DocuMind AI — Your PDF Research Assistant</h1><p>Ask questions about your papers and inspect the evidence behind every response.</p></div>', unsafe_allow_html=True)
st.caption("Answers may be wrong. Verify important details against the cited source excerpts.")
for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("evidence"):
            with st.expander(f"Retrieved evidence ({len(message['evidence'])} passages)"):
                for item in message["evidence"]:
                    st.markdown(f"**{item['filename']} · page {item['page']}**  \nSource `{item['id']}`")
                    st.markdown(f"> {item['text']}")

with st.expander("Try a question"):
    examples = ["Which model or algorithm is used?", "What dataset was used?", "What accuracy was achieved?", "Explain the proposed methodology.", "What are the main findings?", "What limitations are mentioned?"]
    st.write(" · ".join(examples))

question = st.chat_input("Ask a question about the selected PDF(s)…", disabled=not st.session_state.selected_docs)
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the selected PDFs and asking your local model…"):
                response, evidence = ask(store, question, st.session_state.selected_docs, st.session_state.chat[:-1], retrieval_count)
            st.markdown(response)
            if evidence:
                with st.expander(f"Retrieved evidence ({len(evidence)} passages)", expanded=True):
                    for item in evidence:
                        st.markdown(f"**{item['filename']} · page {item['page']}**  \nSource `{item['id']}`")
                        st.markdown(f"> {item['text']}")
            st.session_state.chat.append({"role": "assistant", "content": response, "evidence": evidence})
        except OllamaError as exc:
            st.error(str(exc))
            st.session_state.chat.append({"role": "assistant", "content": str(exc)})

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.chat = [{"role": "assistant", "content": "Conversation cleared. What would you like to learn from your selected PDF(s)?"}]
    st.rerun()
