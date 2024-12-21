from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def load_model_and_tokenizer(model_name="tiiuae/falcon-7b", device="cpu"):
    """
    Load the model and tokenizer, moving the model to the specified device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def generate_text(prompt, model, tokenizer, device, max_length=500):
    """
    Tokenize input and generate text using the specified device.
    """
    # Tokenize the prompt and move it to the target device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output from the model
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    
    # Decode the output tokens into human-readable text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compare_devices(prompt, model_name="tiiuae/falcon-7b"):
    """
    Compare text generation time on CPU and GPU.
    """
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    results = {}
    
    for device in devices:
        print(f"\nRunning on {device.upper()}...")
        
        # Load model and tokenizer for the specified device
        model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device)
        
        # Measure time taken for text generation
        start_time = time.time()
        generated_text = generate_text(prompt, model, tokenizer, device)
        elapsed_time = time.time() - start_time
        
        # Store results
        results[device] = {"generated_text": generated_text, "time": elapsed_time}
        
        # Print the generated text and elapsed time
        print(f"Generated Text ({device}): {generated_text}")
        print(f"Time Taken ({device}): {elapsed_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Set the input prompt and model name
    prompt = "Explain the concept of machine learning in simple terms."
    model_name = "gpt2-large"

    # Run comparison between CPU and GPU
    comparison_results = compare_devices(prompt, model_name=model_name)

    # Final comparison summary
    print("\nComparison Summary:")
    for device, result in comparison_results.items():
        print(f"{device.upper()} - Time Taken: {result['time']:.2f} seconds")
