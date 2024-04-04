import torch
from language_model import TransformerBlockLM
import numpy as np

def train_model(model, text, train_iters=100, eval_iters=10, lr=1e-3, device='cpu'):
    """
    Trains the Transformer model.

    Parameters:
        model (TransformerBlockLM): The Transformer model instance to be trained.
        text (str): The text corpus used for training.
        train_iters (int): The total number of training iterations.
        eval_iters (int): The frequency of evaluation to report average performance.
        lr (float): Learning rate for the optimizer.
        device (str): The device (CPU/GPU) on which to perform training.
    """
    # Prepare the model with the corpus
    model.prep(text)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Lists to store metrics for plotting
    train_losses = []
    perplexities = []

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(f'params {sum([np.prod(p.size()) for p in model_parameters])}')


    for iteration in range(train_iters):
        input_batch, output_batch = model.get_batch(split='train')
        _, ce_loss = model(input_batch, output_batch)

        optimizer.zero_grad(set_to_none=True)
        ce_loss.backward()
        optimizer.step()

        if iteration % eval_iters == 0 or iteration == train_iters - 1:
            metrics = model.eval_loss(eval_iters=1)  # Consider increasing eval_iters for more accuracy
            train_losses.append(metrics['loss'])
            perplexities.append(metrics['perplexity'])
            print(f"Iter {iteration}: Loss = {metrics['loss']:.4f}, Perplexity = {metrics['perplexity']:.2f}")

    return train_losses, perplexities

# Load your text data
with open('WarrenBuffet.txt', 'r') as f:
    text = f.read()

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for computation")


print("-- Training --")

# Instantiate the model
model = TransformerBlockLM(batch_size=256, input_length=32, embed_size=128,
                           sa_multihead_count=8, sa_head_size=128,
                           pos_embed=True, include_mlp=True, device=device)

# Train the model
train_losses, perplexities = train_model(model, text, train_iters=10000, eval_iters=1000, lr=1e-3, device=device)
# After training, save the model state
torch.save(model.state_dict(), 'transformer_language_model.pth')

print("-- Training Completed--")
print()

print("-- Generating text --")

# Generate text
outputs = model.generate(context_token_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
                         max_new_tokens=1000)
print(outputs)
