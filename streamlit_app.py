import streamlit as st
from chatbot import Chatbot

# App title
st.title("Mindguardian - Mental Health Chatbot")

# Function to create new object of chatbot
def initialize_chatbot():
    st.session_state['chatbot'] = Chatbot()

# Function to clear chat memory by creating a new object of chatbot
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    initialize_chatbot()  # Reinitialize the Chatbot instance

# Adding clear history button on sidebar
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Initialize chatbot object
if "chatbot" not in st.session_state:
    st.session_state['chatbot'] = Chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    intro = '''Hello, I am Mindguardian, a mental health counseling chatbot designed to provide professional guidance and support. 
    My purpose is to offer a safe, non-judgmental, and empathetic space for you to explore your thoughts, feelings, and concerns. How can I help you today?'''
    st.session_state.messages.append({"role": "assistant", "content": intro})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your query here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    print(prompt)
    response = st.session_state['chatbot'].run_chatbot(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

