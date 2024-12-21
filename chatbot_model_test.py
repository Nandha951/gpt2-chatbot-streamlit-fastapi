from chatbot_model import get_model_and_tokenizer

# Load model, tokenizer, and device
model, tokenizer, device = get_model_and_tokenizer()

# Generate a response
prompt = "Explain data analytics"
inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Ensure inputs are moved to GPU or CPU
outputs = model.generate(inputs.input_ids, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Bot Response: {response}")
