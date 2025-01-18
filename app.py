import gradio as gr
import torch
import tiktoken
from train import GPT, ModelConfig

# Load the model
def load_model():
    config = ModelConfig()
    model = GPT(config)
    checkpoint = torch.load('best_model.pt', map_location='cpu')
    model.state_dict = checkpoint['model_state_dict']
    model.eval()
    return model

# Cache the model to avoid reloading
MODEL = load_model()
TOKENIZER = tiktoken.get_encoding("gpt2")

def generate_text(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """Generate Shakespeare-style text from a prompt."""
    input_ids = torch.tensor([TOKENIZER.encode(prompt)], dtype=torch.long)
    
    with torch.no_grad():
        output_ids = MODEL.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )[0]
    
    return TOKENIZER.decode(output_ids.tolist())

# Create a more modern Gradio interface using Blocks
with gr.Blocks(
    title="Shakespeare GPT",
    theme=gr.themes.Soft(),
    css=".container { max-width: 800px; margin: auto; }"
) as demo:
    gr.Markdown(
        """
        # 🎭 Shakespeare GPT
        
        Generate Shakespeare-style text using a 124M parameter GPT model trained on Shakespeare's works.
        
        The model was trained on an NVIDIA L4 GPU and achieved a validation loss of 0.156.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Your Prompt",
                placeholder="Enter your prompt here... (e.g. 'ROMEO: ')",
                lines=3
            )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Maximum Length",
                    info="Number of tokens to generate"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Number of tokens to sample from"
                )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Generated Text",
                lines=12,
                show_copy_button=True
            )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["ROMEO: My love for Juliet burns like", 200, 0.7, 50],
            ["HAMLET: To be, or not to be, that is", 200, 0.7, 50],
            ["MACBETH: Double, double toil and trouble,", 200, 0.7, 50],
            ["OTHELLO: O, beware, my lord, of jealousy;", 200, 0.7, 50],
            ["KING LEAR: How sharper than a serpent's tooth", 200, 0.7, 50]
        ],
        inputs=[prompt, max_length, temperature, top_k],
        outputs=output,
        fn=generate_text,
        cache_examples=True
    )
    
    # Add event handler
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_length, temperature, top_k],
        outputs=output
    )
    
    gr.Markdown(
        """
        ### About
        
        This model is a decoder-only transformer with:
        - 12 layers
        - 12 attention heads
        - 768 embedding dimension
        - Flash attention
        - 124M parameters
        
        [View Model Training Logs](https://wandb.ai/macharlasaiteja/shakespeare-gpt/runs/obtjc8b5) | 
        [View Source Code](https://lightning.ai//era/studios/era-session-12/code)
        """
    )

if __name__ == "__main__":
    demo.queue().launch() 