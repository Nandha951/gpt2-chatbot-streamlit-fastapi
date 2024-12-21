from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_and_tokenizer(model_name="gpt2-large"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Check if CUDA (GPU) is available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to the selected device
    model = model.to(device)

    return model, tokenizer, device
