import gradio as gr
import torch
import tiktoken
from train import GPT, ModelConfig

# Load the model
def load_model():
    config = ModelConfig()
    model = GPT(config)
    checkpoint = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Text generation function
def generate_text(prompt, max_length=200, temperature=0.7, top_k=50):
    enc = tiktoken.get_encoding("gpt2")
    model = load_model()
    
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k
    )[0]
    
    return enc.decode(output_ids.tolist())

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Shakespeare GPT")
    gr.Markdown("A GPT model trained on Shakespeare's works. Enter a prompt to generate Shakespeare-style text.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=2)
            max_length = gr.Slider(minimum=10, maximum=500, value=200, step=10, label="Max Length")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
            generate_btn = gr.Button("Generate")
        
        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=10)
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_length, temperature, top_k],
        outputs=output
    )
    
    gr.Examples(
        examples=[
            ["ROMEO: ", 200, 0.7, 50],
            ["HAMLET: To be, or not to be, ", 200, 0.7, 50],
            ["MACBETH: Double, double toil and trouble, ", 200, 0.7, 50]
        ],
        inputs=[prompt, max_length, temperature, top_k],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch() 