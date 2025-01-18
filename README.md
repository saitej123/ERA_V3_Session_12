---
title: Shakespeare GPT
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: false
---

# Shakespeare GPT

A decoder-only transformer model (124M parameters) trained on Shakespeare's works. The model achieved a minimum loss of 0.156 after training on an NVIDIA L4 GPU.

🎭 **Try it out**: [Shakespeare GPT Space](https://huggingface.co/spaces/Saiteja/shakespeare-gpt)

## Training Results

- Best validation loss: 0.156
- Training environment: Lightning.ai Studio with NVIDIA L4 GPU
- Training visualizations: [Weights & Biases Dashboard](https://wandb.ai/macharlasaiteja/shakespeare-gpt/runs/obtjc8b5?nw=nwusermacharlasaiteja)
- Model weights and code: [Lightning.ai Studio](https://lightning.ai//era/studios/era-session-12/code?turnOn=true)

## Model Architecture

- 12 transformer layers
- 12 attention heads
- 768 embedding dimension
- Flash attention for performance
- Dropout (0.2) for regularization
- Bias terms enabled
- Weight decay (0.1)

## Training Configuration

- Batch size: 12 (effective batch size: 60 with gradient accumulation)
- Gradient accumulation steps: 5
- Learning rate: 6e-4 with cosine decay
- Training iterations: 10,000
- Warmup steps: 1,000
- Gradient clipping: 1.0
- AdamW optimizer (β1=0.9, β2=0.95)
- Memory optimizations:
  - TF32 enabled
  - Reduced sequence length (512)
  - Gradient accumulation
  - Optimized batch size

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Sign up for Weights & Biases (wandb.ai) and login:
```bash
wandb login
```

3. Ensure you have the input.txt file in the root directory.

## Training

Run the training script:
```bash
python train.py
```

The script will:
- Initialize a 124M parameter GPT model
- Train on the Shakespeare dataset
- Log metrics to Weights & Biases
- Save checkpoints and the best model
- Generate sample text periodically

## Monitoring

Visit your Weights & Biases dashboard to monitor:
- Training loss
- Validation loss
- Learning rate schedule
- Generated text samples

## Checkpoints

The script saves:
- Regular checkpoints every 1000 iterations in `checkpoint_{iter}.pt`
- Best model based on validation loss in `best_model.pt`

## Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face:
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Set the Space as "Public"

2. Upload the following files to your Space:
   - `train.py`: Model architecture and training code
   - `app.py`: Gradio interface for text generation
   - `requirements.txt`: Dependencies
   - `best_model.pt`: Your trained model checkpoint

3. The Gradio app provides:
   - Interactive text generation interface
   - Adjustable parameters:
     - Max length (10-500 tokens)
     - Temperature (0.1-2.0)
     - Top-k sampling (1-100)
   - Example prompts from Shakespeare's works
   - Real-time text generation

4. The app will automatically deploy and be available at:
   `https://huggingface.co/spaces/[YOUR_USERNAME]/[SPACE_NAME]`

## Links

- Training Dashboard: [Weights & Biases](https://wandb.ai/macharlasaiteja/shakespeare-gpt/runs/obtjc8b5?nw=nwusermacharlasaiteja)
- Model & Code: [Lightning.ai Studio](https://lightning.ai//era/studios/era-session-12/code?turnOn=true)
- GitHub Repository: [shakespeare-gpt](https://github.com/[YOUR_USERNAME]/shakespeare-gpt)
- Live Demo: [Shakespeare GPT Space](https://huggingface.co/spaces/Saiteja/shakespeare-gpt) 