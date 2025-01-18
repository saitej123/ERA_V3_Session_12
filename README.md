# Shakespeare GPT

A decoder-only transformer model (124M parameters) trained on Shakespeare's works.

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

## Model Architecture

- 12 transformer layers
- 12 attention heads
- 768 embedding dimension
- Flash attention for performance
- Cosine learning rate schedule with warmup
- AdamW optimizer with weight decay
- Gradient clipping
- Dropout for regularization

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