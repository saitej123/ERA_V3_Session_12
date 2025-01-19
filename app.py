import gradio.components as gr_components
import gradio.themes as gr_themes
from gradio.blocks import Blocks
from gradio.helpers import Examples
from gradio.components.markdown import Markdown
from gradio.layouts import Row, Column
import torch
import tiktoken
from train import GPT, ModelConfig

# Load the model
def load_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = ModelConfig()
    model = GPT(config)
    checkpoint = torch.load('checkpoint_3000.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

# Cache the model and device to avoid reloading
MODEL, DEVICE = load_model()
TOKENIZER = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = TOKENIZER.n_vocab

def generate_text(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """Generate Shakespeare-style text from a prompt."""
    input_ids = torch.tensor([TOKENIZER.encode(prompt)], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        output_ids = MODEL.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=min(top_k, VOCAB_SIZE)
        )[0]
    
    # Filter out invalid tokens
    valid_tokens = [token for token in output_ids.tolist() if token < VOCAB_SIZE]
    return TOKENIZER.decode(valid_tokens)

# Create a more modern Gradio interface using Blocks
with Blocks(
    title="Shakespeare GPT",
    theme=gr_themes.Soft(),
    css=".container { max-width: 800px; margin: auto; }"
) as demo:
    Markdown(
        """
        # 🎭 Shakespeare GPT
        
        Generate Shakespeare-style text using a 124M parameter GPT model trained on Shakespeare's works.
        
        The model was trained on an NVIDIA L40S GPU and achieved a training loss of 0.064.
        """
    )
    
    with Row():
        with Column(scale=2):
            prompt = gr_components.Textbox(
                label="Your Prompt",
                placeholder="Enter your prompt here... (e.g. 'ROMEO: ')",
                lines=3
            )
            
            with Row():
                max_length = gr_components.Slider(
                    minimum=10,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Maximum Length",
                    info="Number of tokens to generate"
                )
                temperature = gr_components.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                top_k = gr_components.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Number of tokens to sample from"
                )
            
            generate_btn = gr_components.Button("Generate", variant="primary")
        
        with Column(scale=3):
            output = gr_components.Textbox(
                label="Generated Text",
                lines=12,
                show_copy_button=True
            )
    
    # Example prompts
    Examples(
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
    
    Markdown(
        """
        ### About
        
        This model is a decoder-only transformer with:
        - 12 layers
        - 12 attention heads
        - 768 embedding dimension
        - Flash attention
        - 124M parameters
        
        [View Model Training Logs](https://wandb.ai/macharlasaiteja/shakespeare-gpt/runs/3pr6gpfk?nw=nwusermacharlasaiteja) | 
        [View Source Code](https://lightning.ai/saitej/era/studios/era-session-12/web-ui)
        """
    )

if __name__ == "__main__":
    demo.queue().launch() 