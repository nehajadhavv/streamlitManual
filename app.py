import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded correctly
if not groq_api_key:
    st.error("API Key is missing! Please check your .env file.")
    st.stop()

st.title("HCMNEXT FAQ Chatbot")

# Default model setup
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "mixtral-8x7b-32768"  # Default Groq model

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("Ask me anything...")
if prompt:
    # Append user input to the message history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant's response using Groq API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            groq_chat = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=st.session_state["groq_model"]
            )

            response = groq_chat.invoke(prompt)  # Invoke Groq API

            full_response = response.content  # Get response text
            message_placeholder.markdown(full_response)  # Display final response

            # Append assistant's response to the message history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {e}")
