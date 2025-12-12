import os
import re
import streamlit as st
from transformers import pipeline

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="LLM Chatbot Demo (CTTE)", page_icon="🤖", layout="wide")

# ------------------------------
# MODEL CONFIG
# ------------------------------
MODEL_CONFIG = {
    "google/flan-t5-small": {
        "task": "text2text-generation",
        "description": "FLAN-T5 Small: instruction-following seq2seq model. Good for Q&A, summarization, rewriting.",
    },
    "distilgpt2": {
        "task": "text-generation",
        "description": "DistilGPT2: small causal LM. Good for continuation-style generation.",
    },
}

DOCS_DIR = "docs"


# ------------------------------
# HELPERS: RAG (simple keyword overlap)
# ------------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return set(toks)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def build_rag_index(selected_files: list[str], chunk_size: int, overlap: int) -> list[dict]:
    """
    Returns list of dicts: {"source": filename, "chunk": chunk}
    """
    index = []
    for fname in selected_files:
        fpath = os.path.join(DOCS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        for ch in chunk_text(content, chunk_size=chunk_size, overlap=overlap):
            index.append({"source": fname, "chunk": ch})
    return index


def retrieve_top_k(index: list[dict], query: str, k: int) -> list[dict]:
    if not index:
        return []
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scored = []
    for item in index:
        c_tokens = tokenize(item["chunk"])
        score = len(q_tokens.intersection(c_tokens))
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]


def format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""
    parts = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(f"[Source: {r['source']} | Chunk {i}]\n{r['chunk']}")
    return "\n\n---\n\n".join(parts)


def build_prompt(user_msg: str, rag_context: str, model_name: str) -> str:
    """
    Builds a prompt appropriate for both instruction models and causal LMs.
    """
    if rag_context:
        base = (
            "You are a helpful assistant. Use the provided context to answer. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"CONTEXT:\n{rag_context}\n\n"
            f"QUESTION:\n{user_msg}\n\n"
            "ANSWER:"
        )
    else:
        base = user_msg

    # For GPT2-like models, add small formatting cues
    if model_name == "distilgpt2":
        base = f"User: {user_msg}\n" + (f"Context:\n{rag_context}\n" if rag_context else "") + "Assistant:"
    return base


# ------------------------------
# PIPELINE LOADER (cached)
# ------------------------------
@st.cache_resource
def load_pipe(model_name: str, task: str):
    return pipeline(task, model=model_name)


# ------------------------------
# SESSION STATE INIT
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_index" not in st.session_state:
    st.session_state.rag_index = []

if "rag_files" not in st.session_state:
    st.session_state.rag_files = []


# ------------------------------
# SIDEBAR UI
# ------------------------------
st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox("Choose a model", list(MODEL_CONFIG.keys()))
st.sidebar.caption(MODEL_CONFIG[model_name]["description"])

gen_max_new_tokens = st.sidebar.slider("max_new_tokens", 32, 256, 128, 8)
gen_temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.7, 0.1)
gen_top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)

st.sidebar.divider()
enable_rag = st.sidebar.checkbox("✅ Enable RAG (from workspace docs/ folder)", value=False)

# Defaults
selected_docs = []
chunk_size = 800
chunk_overlap = 150
top_k = 3

if enable_rag:
    st.sidebar.markdown("### 📁 Documents (CodeSandbox-safe)")

    if not os.path.exists(DOCS_DIR):
        st.sidebar.warning("No 'docs/' folder found. Create it in the project root.")
    else:
        txt_files = sorted([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")])
        if not txt_files:
            st.sidebar.info("No .txt files in docs/. Add some .txt files to use RAG.")
        else:
            selected_docs = st.sidebar.multiselect(
                "Select .txt files to index",
                txt_files,
                default=st.session_state.rag_files if st.session_state.rag_files else txt_files[:1],
            )

            chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 2000, 800, 50)
            chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 0, 500, 150, 10)
            top_k = st.sidebar.slider("Top-K chunks to retrieve", 1, 8, 3, 1)

            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                if st.button("🔄 Rebuild RAG index"):
                    st.session_state.rag_index = build_rag_index(selected_docs, chunk_size, chunk_overlap)
                    st.session_state.rag_files = selected_docs
                    st.sidebar.success(f"Indexed {len(st.session_state.rag_index)} chunks.")
            with col_b:
                if st.button("🧹 Clear RAG index"):
                    st.session_state.rag_index = []
                    st.session_state.rag_files = []
                    st.sidebar.info("Cleared.")

    st.sidebar.caption("Tip: Add .txt files into `docs/` via the file tree or terminal. Avoid Streamlit upload in CodeSandbox.")


# ------------------------------
# MAIN UI
# ------------------------------
st.title("🤖 LLM Chatbot Demo (with CodeSandbox-safe RAG)")
st.caption("Model switching + parameter tuning + optional RAG from workspace `docs/` folder (no file uploader).")

# Always define retrieved so debug expander never errors
retrieved: list[dict] = []

st.subheader("💬 Chat")

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Type your message...")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    cfg = MODEL_CONFIG[model_name]
    pipe = load_pipe(model_name, cfg["task"])

    # RAG retrieval (if enabled + index exists)
    rag_context = ""
    retrieved = []
    if enable_rag and st.session_state.rag_index:
        retrieved = retrieve_top_k(st.session_state.rag_index, user_msg, k=top_k)
        rag_context = format_context(retrieved)

    prompt = build_prompt(user_msg, rag_context, model_name)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            try:
                if cfg["task"] == "text2text-generation":
                    out = pipe(
                        prompt,
                        max_new_tokens=gen_max_new_tokens,
                        do_sample=True,
                        temperature=gen_temperature,
                        top_p=gen_top_p,
                    )[0]["generated_text"]
                    assistant_text = out.strip()
                else:
                    out = pipe(
                        prompt,
                        max_new_tokens=gen_max_new_tokens,
                        do_sample=True,
                        temperature=gen_temperature,
                        top_p=gen_top_p,
                        pad_token_id=pipe.tokenizer.eos_token_id,
                    )[0]["generated_text"]
                    assistant_text = out.replace(prompt, "").strip() or out.strip()

            except Exception as e:
                assistant_text = f"⚠️ Error while generating: {e}"

        st.markdown(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})


# ------------------------------
# RAG DEBUG (EXPANDER)
# ------------------------------
if enable_rag:
    with st.expander("🧪 RAG Debug (for learning)", expanded=False):
        if not st.session_state.rag_index:
            st.warning("RAG enabled but index is empty. Select docs and click **Rebuild RAG index**.")
        else:
            st.write(f"**Indexed chunks:** {len(st.session_state.rag_index)}")
            st.write(f"**Indexed files:** {', '.join(st.session_state.rag_files) if st.session_state.rag_files else '—'}")

            if retrieved:
                st.markdown("### Retrieved Chunks (this question)")
                for r in retrieved:
                    st.markdown(f"**Source:** {r['source']}")
                    st.write(r["chunk"][:700] + ("..." if len(r["chunk"]) > 700 else ""))
                    st.divider()


st.divider()

# ------------------------------
# UTILITIES
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("📁 Create sample docs/ files (demo)"):
        os.makedirs(DOCS_DIR, exist_ok=True)
        samples = {
            "student_profile.txt": (
                "Student Profile:\n"
                "Name: Aditi Kumar\n"
                "College: CTTE Women's College\n"
                "Interests: AI, analytics, and product management.\n"
            ),
            "rag_basics.txt": (
                "Retrieval-Augmented Generation (RAG) combines information retrieval with LLM generation.\n"
                "Documents are split into chunks, embedded, and stored in a vector index.\n"
                "At query time, relevant chunks are retrieved and added to the prompt.\n"
            ),
            "llm_params.txt": (
                "Common generation parameters:\n"
                "- max_new_tokens: limits response length\n"
                "- temperature: controls randomness (higher = more random)\n"
                "- top_p: nucleus sampling; keeps the smallest set of tokens whose cumulative prob >= top_p\n"
            ),
        }
        for fname, text in samples.items():
            with open(os.path.join(DOCS_DIR, fname), "w", encoding="utf-8") as f:
                f.write(text)
        st.success("Created sample .txt files in docs/. Enable RAG and rebuild index.")
