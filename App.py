import os
import streamlit as st

# ------------------------------
# PAGE SETUP
# ------------------------------
st.set_page_config(page_title="CTTE RAG Demo", page_icon="📄")
st.title("📄 Document Loader Demo (CodeSandbox Safe)")

st.markdown(
    """
This app loads **text files from the project workspace**.
Upload via browser is disabled to avoid CodeSandbox errors.
"""
)

# ------------------------------
# DOCUMENT DIRECTORY
# ------------------------------
DOCS_DIR = "docs"

if not os.path.exists(DOCS_DIR):
    st.warning("📂 'docs/' folder not found. Create it and add .txt files.")
    st.stop()

# ------------------------------
# LIST TEXT FILES
# ------------------------------
txt_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]

if not txt_files:
    st.info("No .txt files found in docs/. Add some files to continue.")
    st.stop()

selected_file = st.selectbox(
    "Select a document to load:",
    txt_files
)

# ------------------------------
# READ FILE CONTENT
# ------------------------------
file_path = os.path.join(DOCS_DIR, selected_file)

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

st.subheader("📄 File Content")
st.text_area(
    "Loaded text:",
    content,
    height=300
)

# ------------------------------
# PLACEHOLDER FOR RAG
# ------------------------------
st.divider()
st.subheader("🔍 (Next Step) RAG / Q&A")

user_question = st.text_input(
    "Ask a question about this document:"
)

if user_question:
    st.info("This is where retrieval + LLM answering will go in Session 4.")
