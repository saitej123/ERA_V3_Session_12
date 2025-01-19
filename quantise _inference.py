import torch
import torch.nn as nn
import torch.ao.quantization
from train import GPT, ModelConfig
import os

def quantize_model(input_path='checkpoint_3000.pt', output_path='quantized_model.pt'):
    print("Loading model...")
    config = ModelConfig()
    model = GPT(config)
    
    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Original model size:", os.path.getsize(input_path) / (1024 * 1024 * 1024), "GB")
    
    # Use dynamic quantization for linear layers only
    print("Quantizing model to int8...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize linear layers
        dtype=torch.qint8
    )
    
    # Save the quantized model
    print("Saving quantized model...")
    torch.save({
        'state_dict': quantized_model.state_dict(),
        'config': config
    }, output_path)
    
    print("Quantized model size:", os.path.getsize(output_path) / (1024 * 1024 * 1024), "GB")
    return quantized_model

if __name__ == "__main__":
    try:
        quantized_model = quantize_model()
        print("Model quantization complete!")
    except Exception as e:
        print(f"Error during model quantization: {e}") 