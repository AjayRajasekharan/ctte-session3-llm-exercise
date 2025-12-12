import streamlit as st
from transformers import pipeline

# ------------------------------
# Run Streamlit app
# ------------------------------
# To run streamlit app use the below command in bash terminal:
# streamlit run Demo/lllm_advance_demo.py

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(page_title="LLM Chatbot Demo", page_icon="🤖")

# ------------------------------
# MODEL CONFIGURATION
# ------------------------------
# We EXPLICITLY map each model to a pipeline task.
# Students only pick the model; task selection is internal.
#
# For developers:
# - Each model has a primary "task" used in this demo.
# - Under each model, we list other possible model–task combinations
#   as comments for future experiments.

MODEL_CONFIG = {
    "google/flan-t5-small": {
        # Main task used in this demo:
        "task": "text2text-generation",
        "description": "FLAN-T5 SMALL: instruction-following seq2seq model. Good for Q&A, translation, summarization.",
        # Other possible model–task combinations (NOT used in this app):
        # - pipeline("text2text-generation", model="google/flan-t5-small")
        # - pipeline("summarization", model="google/flan-t5-small")
        # - pipeline("translation_en_to_de", model="google/flan-t5-small")
    },
    "google/flan-t5-base": {
        "task": "text2text-generation",
        "description": "FLAN-T5 BASE: larger FLAN-T5. Better quality, heavier to run (more RAM/CPU).",
        # Other possible model–task combinations:
        # - pipeline("text2text-generation", model="google/flan-t5-base")
        # - pipeline("summarization", model="google/flan-t5-base")
        # - pipeline("translation_en_to_fr", model="google/flan-t5-base")
    },
    # "google/flan-t5-large": {
    #     "task": "text2text-generation",
    #     "description": "FLAN-T5 LARGE: larger FLAN-T5. Better quality, heavier to run (more RAM/CPU).",
    #     # Other possible model–task combinations:
    #     # - pipeline("text2text-generation", model="google/flan-t5-large")
    #     # - pipeline("summarization", model="google/flan-t5-large")
    #     # - pipeline("translation_en_to_fr", model="google/flan-t5-large")
    # },
    "distilgpt2": {
        "task": "text-generation",
        "description": "DISTILGPT2: tiny GPT-2 style model. Good for generic text continuation.",
        # Other possible model–task combinations:
        # - pipeline("text-generation", model="distilgpt2")  # main use-case
        # GPT-style models are almost always used with text-generation.
    },
    "microsoft/DialoGPT-small": {
        "task": "text-generation",
        "description": "DIALOGPT SMALL: dialogue-tuned GPT-2. Good for chat-style replies.",
        # Other possible model–task combinations:
        # - pipeline("text-generation", model="microsoft/DialoGPT-small")  # what we use here
        # - pipeline("conversational", model="microsoft/DialoGPT-small")
        #   (requires Hugging Face `Conversational` objects – more advanced)
    },

    # --------------------------------
    # EXTRA MODELS (EXAMPLES FOR LATER)
    # --------------------------------
    # To add these, uncomment the block(s) below and they will appear
    # automatically in the model dropdown.
    #
    # "facebook/bart-base": {
    #     "task": "text2text-generation",
    #     "description": "BART base: seq2seq model for summarization and general text-to-text tasks.",
    #     # Other model–task combinations:
    #     # - pipeline("summarization", model="facebook/bart-base")
    #     # - pipeline("text2text-generation", model="facebook/bart-base")
    # },
}

MODEL_NAMES = list(MODEL_CONFIG.keys())

# ------------------------------
# PROMPT EXAMPLES (SIDEBAR REFERENCE)
# ------------------------------
# These are "best effort" examples to guide students on how to prompt each model.
# They are shown in the sidebar and do NOT change app logic.

PROMPT_EXAMPLES = {
    "google/flan-t5-small": [
        "Explain cloud computing in simple terms:",
        "Explain step by step how linear regression works:",
        "Explain in bullet points: advantages of Python",
        "Summarize the following text:\n<PASTE TEXT HERE>",
        "Rewrite in simple language:\n<PASTE TEXT HERE>",
        "Translate English to German: How are you today?",
        "Compare REST vs GraphQL:",
    ],
    "google/flan-t5-base": [
        "Explain cloud computing in simple terms:",
        "Explain step by step how gradient descent works:",
        "Explain in bullet points: supervised vs unsupervised learning",
        "Summarize the following text:\n<PASTE TEXT HERE>",
        "Extract the key takeaways from the following text:\n<PASTE TEXT HERE>",
        "Translate English to French: Where is the nearest train station?",
        "Give 3 examples of classification problems:",
    ],
    "distilgpt2": [
        "Cloud computing is a technology that allows",
        "Machine learning is important because",
        "Once upon a time, a data scientist",
        "The advantages of using Python include",
        "In recent years, artificial intelligence has",
    ],
    "microsoft/DialoGPT-small": [
        "Hi! Can you explain cloud computing?",
        "What do you think about machine learning?",
        "Can you help me understand neural networks?",
        "Why is Python so popular?",
        "That makes sense—can you explain more?",
    ],
}

# ------------------------------
# MODEL LOADER (CACHED)
# ------------------------------

@st.cache_resource
def load_model(model_name: str):
    """
    Load a Hugging Face pipeline for the chosen model.

    Steps:
      - Look up the correct task from MODEL_CONFIG
      - Create pipeline(task, model=model_name)
      - Cache the result so each model is loaded only once
    """
    cfg = MODEL_CONFIG[model_name]
    task = cfg["task"]
    pipe = pipeline(task, model=model_name)
    return pipe, task


# ------------------------------
# SIDEBAR: SETTINGS
# ------------------------------

st.title("🤖 LLM Chatbot Demo")

with st.sidebar:
    st.header("Settings")

    # 1) Choose model
    model_choice = st.selectbox(
        "Choose a model",
        MODEL_NAMES,
        help=(
            "flan-t5-small / flan-t5-base → better for Q&A, instructions.\n"
            "distilgpt2 → tiny text generator.\n"
            "DialoGPT → chatty conversation model."
        ),
    )

    st.markdown(f"**Model description:** {MODEL_CONFIG[model_choice]['description']}")

    # 2) Max new tokens
    max_new_tokens = st.slider(
        "Max new tokens",
        16, 256, 128, 16,
        help="How long the answer can be. More tokens = longer / slower response."
    )

    # 3) Temperature (commented out for now – can be enabled in future sessions)
    # temperature = st.slider(
    #     "Creativity (temperature)",
    #     0.1, 1.5, 0.7, 0.1,
    #     help="Lower = more deterministic, Higher = more random/creative."
    # )

    # ------------------------------
    # Prompt examples reference section
    # ------------------------------
    with st.expander("🧠 Best prompt examples (model-specific)"):
        st.markdown(
            "**Quick mental model:**\n"
            "- **FLAN-T5** = task follower (use *instruction + colon*)\n"
            "- **distilgpt2** = autocomplete (start a sentence)\n"
            "- **DialoGPT** = chatty (ask casually)\n"
        )

        examples = PROMPT_EXAMPLES.get(model_choice, [])
        if examples:
            st.markdown(f"**Examples for `{model_choice}`:**")
            for ex in examples:
                st.code(ex)
        else:
            st.markdown("No examples available for this model yet.")

    st.markdown("---")

    # 4) Clear chat history button
    if st.button("🧹 Clear chat history"):
        # Remove the messages list from session state and rerun the app,
        # which resets the visible chat history.
        st.session_state.pop("messages", None)
        st.experimental_rerun()


# ------------------------------
# LOAD MODEL + TASK
# ------------------------------

pipe, current_task = load_model(model_choice)

# Optional: small caption so students see which task is being used
st.caption(f"Using model: `{model_choice}` with pipeline task: `{current_task}`")


# ------------------------------
# CHAT HISTORY (SESSION STATE)
# ------------------------------

# Initialize chat history (frontend memory) if not present
if "messages" not in st.session_state:
    # Each entry: {"role": "user" | "assistant", "content": str}
    st.session_state.messages = []

# Display previous messages from this session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------
# RESPONSE GENERATION HELPER
# ------------------------------
def generate_response(
    pipe,
    model_name: str,
    user_input: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Call the pipeline and post-process the output.

    - Uses do_sample=True so students can see some variation.
    - For GPT-style models (distilgpt2 / DialoGPT), strips echoed user input
      from the beginning of the generated text.
    """

    # Define prompt BEFORE using it
    prompt_text = user_input

    outputs = pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    # pipelines return: [{"generated_text": "..."}]
    text = outputs[0].get("generated_text", "").strip()

    if not text:
        return "I couldn't generate a response. Please try rephrasing your question."

    # Post-process GPT-style models: they often echo the user input
    if model_name in ["distilgpt2", "microsoft/DialoGPT-small"]:
        if text.lower().startswith(user_input.lower()):
            text = text[len(user_input):].strip()

        if not text:
            return "Could you try asking in a slightly different way?"

    return text


# ------------------------------
# MAIN CHAT LOOP
# ------------------------------
prompt = st.chat_input("Ask a classroom-safe question:")

if prompt:
    # 1) Save user message (UI history)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) Show the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3) Generate + show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(
                pipe,
                model_name=model_choice,
                user_input=prompt,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
            )
            st.markdown(response)

    # 4) Save assistant response (UI history)
    st.session_state.messages.append({"role": "assistant", "content": response})
