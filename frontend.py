import streamlit as st
import requests

# Streamlit app title
st.title("Customer Support Chatbot")

# Input for user prompt
prompt = st.text_input("Enter your query:")

if st.button("Send"):
    try:
        # Send the prompt to the FastAPI server and retrieve response
        response = requests.post(
            "http://127.0.0.1:8000/chat",  # Ensure this URL matches your FastAPI server setup
            json={"prompt": prompt}
        ).json()

        # Display the bot's response
        st.text(f"Bot Response: {response['response']}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the server: {e}")
