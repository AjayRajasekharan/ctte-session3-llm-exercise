import streamlit as st
from transformers import pipeline

# ------------------------------
# SETUP
# ------------------------------

# Configure Streamlit page
st.set_page_config(page_title="LLM Chatbot (Model Choice)")

@st.cache_resource
def load_model(model_name: str = "google/flan-t5-small"):
    """
    Load a Hugging Face model based on its name and task.
    Uses a dictionary mapping to make adding models easier.

    Note:
    We use @st.cache_resource so the model loads only once and is stored in memory. 
    Without this, the model would reload every time the app refreshes (making it very slow). 
    """

    # Define which task each model uses
    model_tasks = {
        "google/flan-t5-small": "text2text-generation",   # Q&A / instructions
        "distilgpt2": "text-generation",                  # generic text gen
        "microsoft/DialoGPT-small": "text-generation",    # dialogue/chat
        # Add more models here if needed and update UI
        # "facebook/bart-base": "text2text-generation",
    }

    # Get the task for this model (default = text-generation)
    task = model_tasks.get(model_name, "text-generation")

    return pipeline(task, model=model_name)

# ------------------------------
# UI LAYOUT
# ------------------------------

st.title("LLM Chatbot with Model Choice")

# Sidebar controls → let the student pick a model and answer length
with st.sidebar:
    # Dropdown menu for model selection
    model_choice = st.selectbox(
        "Choose a model",
        ["google/flan-t5-small", "distilgpt2", "microsoft/DialoGPT-small"],
        help="flan-t5-small = better for Q&A, distilgpt2 = tiny text generator, DialoGPT = chatty model"
    )

    # Slider for answer length (max number of tokens)
    max_new_tokens = st.slider(
        "Max new tokens",
        16, 256, 128, 16,
        help="Controls how long the answer can be (more tokens = longer answers)"
    )

# Load the model once (cached)
pipe = load_model(model_choice)

# ------------------------------
# MAIN CHAT INPUT
# ------------------------------

# Textbox for user input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:", key="input_text")
    go = st.form_submit_button("Ask") # Triggered on Enter or click

if go and user_input.strip():
    st.write("User:", user_input)
    with st.spinner("Thinking..."):
        # Generate response using the selected model
        out = pipe(user_input, max_new_tokens=max_new_tokens, do_sample=True)

        # Show the response
        st.write("Bot:", out[0]["generated_text"])