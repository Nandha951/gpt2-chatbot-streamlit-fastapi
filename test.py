from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_and_tokenizer(model_name="gpt2-large"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Check if CUDA (GPU) is available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move the model to GPU or CPU
    model = model.to(device)
    
    return model, tokenizer, device

def generate_text(prompt, model, tokenizer, device, max_length=50):
    # Tokenize the input and move tokens to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text using the model
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
if __name__ == "__main__":
    model_name = "gpt2-large"
    model, tokenizer, device = get_model_and_tokenizer(model_name=model_name)
    
    prompt = "Explain data analytics"
    generated = generate_text(prompt, model, tokenizer, device)
    
    model_name = "gpt2-large"
    model, tokenizer, device = get_model_and_tokenizer(model_name=model_name)
    
    prompt = "Explain data analytics"
    generated = generate_text(prompt, model, tokenizer, device)
    print("Generated Text:")
    print(generated)
