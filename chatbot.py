# app.py
import json
import time
from datetime import datetime
import streamlit as st
from transformers import pipeline

# page config 
st.set_page_config(
    page_title="Morris – SE Chatbot",
    page_icon="🤖",
    layout="centered"
)

# session state initialization 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dark" not in st.session_state:
    st.session_state.dark = False

# a simple theming mechanism 

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

# sidebar controls 

with st.sidebar:
    st.title("⚙️ Controls")
    max_new = st.slider("Max new tokens", 20, 1_000, 150, 10)
    temp = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    dark_mode = st.checkbox("Dark mode", value=dark)
    if dark_mode != dark:
        st.session_state.dark = dark_mode
        st.rerun()
    if st.button("🗑️  Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    if st.download_button(
        label="💾 Download chat",
        data=json.dumps(st.session_state.chat_history, indent=2),
        file_name=f"chat_{datetime.now():%Y-%m-%d_%H-%M}.json",
        mime="application/json",
    ):
        st.success("Downloaded!")

# main title

st.title("🤖 Morris – SE Chatbot")
st.caption("Ask me anything about software engineering.")

# load model (cached)

@st.cache_resource(show_spinner=False)
def load_model():
    pipe = pipeline("text-generation", model="gpt2")
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    return pipe

generator = load_model()

# prompt builder 

SYSTEM_PROMPT = (
    "You are a helpful assistant for software engineering. "
    "Answer concisely, use markdown for formatting and fenced code blocks for code."
)

def build_prompt(history, question):
    lines = [SYSTEM_PROMPT]
    for hq, ha in history[-6:]:          # keep last 6 turns
        lines.append(f"User: {hq}")
        lines.append(f"Assistant: {ha}")
    lines.append(f"User: {question}")
    lines.append("Assistant:")
    return "\n".join(lines)

# display chat

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# user input handling

if prompt := st.chat_input("Type your message here…"):
    # append user message
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full = ""
        prompt_text = build_prompt(st.session_state.chat_history, prompt)
        # stream tokens manually
        for new_text in generator(
            prompt_text,
            max_new_tokens=max_new,
            temperature=temp,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"][len(prompt_text) :].split(" "):
            full += new_text + " "
            message_placeholder.markdown(full + "▌")
            time.sleep(0.02)
        message_placeholder.markdown(full)

    # store assistant answer
    st.session_state.chat_history.append(("assistant", full.strip()))