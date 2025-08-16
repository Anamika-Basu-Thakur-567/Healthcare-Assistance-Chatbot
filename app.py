import os
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

if not HF_TOKEN or not HF_MODEL:
    st.error("Missing HF_TOKEN or HF_MODEL in your .env file")
    st.stop()

client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

st.set_page_config(page_title="HealthAssist AI", page_icon="💬", layout="centered")
st.title("🩺 HealthAssist AI")
st.write("Ask me any health-related question. I am **not a doctor** — please consult professionals for medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = ""
            placeholder = st.empty()

            # Build full message history including system prompt + previous chat
            messages = [
                {"role": "system", "content": "You are a helpful health assistant. Avoid giving medical advice, provide only general health information."}
            ]
            messages.extend(st.session_state.messages)  # includes current user message too

            try:
                for message in client.chat_completion(
                    messages=messages,
                    max_tokens=500,
                    stream=True
                ):
                    if message.choices[0].delta.get("content"):
                        token = message.choices[0].delta["content"]
                        response_text += token
                        placeholder.markdown(response_text)

                # Save assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Error: {str(e)}")




