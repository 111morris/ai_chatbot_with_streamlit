# app.py
import json
import time
from datetime import datetime
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


# PAGE CONFIG AND SESSION STATE

st.set_page_config(
    page_title="Morris – SE Chatbot",
    page_icon="🤖",
    layout="centered"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "dark" not in st.session_state:
    st.session_state.dark = False

# THEME SETUP

dark = st.session_state.dark
theme = {
    "bg": "#0e1117" if dark else "#ffffff",
    "fg": "#ffffff" if dark else "#000000",
    "secondary": "#262730" if dark else "#f0f2f6",
}

st.markdown(
    f"""
    <style>
    .stApp {{background-color: {theme["bg"]}; color: {theme["fg"]};}}
    .css-1d391kg, .css-18e3th9 {{background-color: {theme["secondary"]};}}
    </style>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR CONTROLS

with st.sidebar:
    st.title("Controls")
    max_new = st.slider("Max new tokens", 20, 800, 150, 10)
    temp = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)

    dark_mode = st.checkbox("Dark mode", value=dark)
    if dark_mode != dark:
        st.session_state.dark = dark_mode
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.download_button(
        label="Download chat",
        data=json.dumps(st.session_state.chat_history, indent=2),
        file_name=f"chat_{datetime.now():%Y-%m-%d_%H-%M}.json",
        mime="application/json",
    )

# title 

st.title("Morris SE Chatbot")
st.caption("Ask me anything about software engineering.")

# load model (cached)
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    return tokenizer, model


pipe, tokenizer = load_model()

# chat template builder 

def build_prompt(chat_history, user_input):
    msgs = [{"role": "system", "content": 
             "You are a helpful assistant skilled in software engineering. "
             "Answer clearly and use markdown when necessary."}]

    for role, msg in chat_history:
        msgs.append({"role": role, "content": msg})

    msgs.append({"role": "user", "content": user_input})

    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

# display chat histroy 

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# handling user input 

if prompt := st.chat_input("Type your message…"):

    # Store user message
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build prompt
    prompt_text = build_prompt(st.session_state.chat_history, prompt)

    # Generate response
    with st.chat_message("assistant"):
        msg_box = st.empty()
        full_response = ""

        outputs = pipe(
            prompt_text,
            max_new_tokens=max_new,
            temperature=temp,
            do_sample=True,
            return_full_text=False,
        )

        # Stream token by token
        tokens = outputs[0]["generated_text"]
        for t in tokens.split():
            full_response += t + " "
            msg_box.markdown(full_response + "▌")
            time.sleep(0.02)

        msg_box.markdown(full_response)

    # Store assistant response
    st.session_state.chat_history.append(("assistant", full_response.strip()))