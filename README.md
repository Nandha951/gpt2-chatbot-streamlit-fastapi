# Chatbot with Hugging Face Transformers, FastAPI, Streamlit, and GPU Support

## Overview

This project involves building a conversational AI chatbot powered by Hugging Face's transformers library with models like GPT-2. The application consists of four parts:

1. **Model Loading (CUDA GPU Support)**: A function to load the GPT-2 model and tokenizer, and determine whether to run on GPU or CPU.
2. **Chat Execution**: Using the loaded model and tokenizer to process user input and generate a response.
3. **Streamlit Application**: A web-based frontend to interact with the chatbot.
4. **FastAPI Backend**: A server that handles the chatbot logic and provides responses via API requests.

---

## Requirements

-   **Python 3.7+**
-   **Packages**:
    -   `transformers`
    -   `torch` (for CUDA GPU support)
    -   `streamlit` (for frontend)
    -   `fastapi` and `uvicorn` (for backend)

Ensure you have the necessary packages by installing them via:

```bash
pip install -r requirements.txt
```
