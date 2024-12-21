from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_model import get_model_and_tokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model, tokenizer, and device
model, tokenizer, device = get_model_and_tokenizer()

class Request(BaseModel):
    prompt: str

@app.post("/chat")
def chat(req: Request):
    # Tokenize user input and move inputs to device (CPU/GPU)
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    
    # Generate model output
    outputs = model.generate(inputs.input_ids, max_length=50)
    
    # Decode response and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
