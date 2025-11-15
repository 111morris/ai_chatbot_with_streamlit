import streamlit as st
from tranformers import pipeline 

st.set_page_config(page_title="Chatbot")

def load_text_generator(): 
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generatoru

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for software Engineering",
    "Answer concisely and to the point",
    "Use markdown to format your answers",
    "Use code blocks to format your answers"
)

# Build the convo prompt 

def build_conversation_prompt(chat_history, user_quesion):
    formated_conversation = []
    for previous_question, previous_answer in chat_history: 
        formated_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}")
        
    formated_conversation.append(f"User: {user_quesion}\nAssistant:")
    return SYSTEM_INSTRUCTION + "\n" + "\n".join(formated_conversation)


st.title("Morris - ChatBot UI")
st.caption("Ask me anything about software Engineering")

# Sidebar for chat history 
 
with st.sidebar:
    st.header("Model Controls/Config")
    max_new_tokens = st.slider("Max New Tokens", min_value=20, max_value=1000, value=50, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = ["Start new chart"]
        st.success("Chat history cleared")
        
# Display chat history 
for user_message, ai_reply in st.session_state.chat_history:
    st.chat_message("user").markdown(user_message)
    st.chat_message("assistant").markdown(ai_reply)

# User input 
user_input = st.chat_input("Type your message here...")

if user_input:
    st.chat_message("user").markdown(user_input)

    with st.spinner("Generating response..."):
        text_generator = load_text_generator(st.session_state.chat_history, user_input)

        generation_output = text_generator(
            prompt_text, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=temperature,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            eos_token_id=text_generator.tokenizer.eos_token_id,
        )[0]['generated_text']
    
    # Build the prompt 
    prompt = build_conversation_prompt(st.session_state.chat_history, user_input)
    
    # Load model and generate response 
    text_generator = load_text_generator()
    response = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=text_generator.tokenizer.eos_token_id
    )[0]['generated_text']
    
    # Extract the assistant's reply from the response
    if "Assistant:" in generation_output:
        generated_answer = generation_output.split("Assistant:")[0].strip()
    
    # Display assistant's reply 
    st.chat_message("assistant").markdown(generated_answer)
    st.session_state.chat_history.append((user_input, generated_answer))
    
    # Update chat history 
    st.session_state.chat_history.append((user_input, assistant_reply))