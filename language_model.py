import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformer_blocks import TransformerBlock
import random

class TransformerBlockLM(nn.Module):
    """
    This class implements a Transformer-based language model. It encapsulates the entire model
    architecture, including self-attention heads, multi-head attention, and position-wise feedforward
    networks within transformer blocks. It's designed to be flexible, allowing for adjustments to
    various hyperparameters like batch size, input length, embedding size, and more.

    Attributes:
        device (torch.device): The device (CPU/GPU) on which the model will run.
        blocks (nn.Sequential): A sequence of Transformer blocks forming the model's backbone.
        Various other attributes for model configuration and data handling.
    """
    def __init__(self, batch_size=4, input_length=8, embed_size=16, sa_head_size=8,
                 sa_multihead_count=4, pos_embed=False, include_mlp=False, device='cpu'):
        """
        Initializes the TransformerBlockLM model with specified configurations.

        Parameters:
            batch_size (int): The number of sequences per batch.
            input_length (int): The length of input sequences.
            embed_size (int): Dimensionality of token embeddings.
            sa_head_size (int): Dimensionality of each self-attention head's output.
            sa_multihead_count (int): Number of heads in the multi-head attention mechanism.
            pos_embed (bool): If True, use positional embeddings.
            include_mlp (bool): If True, include a multi-layer perceptron in each Transformer block.
            device (str): The device (CPU/GPU) for tensors and model components.
        """
        super().__init__()
        self.device = torch.device(device)  # Set device according to user input
        self.blocks = None
        self.ffn = None
        self.sa_heads = None
        # sa_head_size head_size of self-attention module
        self.sa_head_size = sa_head_size
        self.sa_multihead_count = sa_multihead_count

        self.val_data = None
        self.train_data = None
        self.val_text = None
        self.train_text = None
        self.K = None
        self.linear_sahead_to_vocab = None
        self.vocab = None
        self.token_embeddings_table = None
        self.vocab_size = None
        self.encoder = None
        self.decoder = None
        self.vocab_size: int
        self.is_pos_emb = pos_embed
        self.include_mlp = include_mlp
        # input_length = how many consecutive tokens/chars in one input
        self.input_length = input_length
        # batch_size = how many inputs are going to be processed in-parallel (on GPU)
        self.batch_size = batch_size
        # embed_size = embedding size
        self.embed_size = embed_size

        self.lm_head = None
        self.position_embeddings_table = None

    def forward(self, in_ids, target=None):
        """
        Defines the forward pass of the model.

        Parameters:
            in_ids (torch.Tensor): Input tensor containing sequences of token IDs.
            target (torch.Tensor, optional): Target tensor for the input sequences.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the logits of the model's
            predictions and, if targets are provided, the cross-entropy loss.
        """

        in_ids_emb = self.token_embeddings_table(in_ids[:, -self.input_length:])
        if self.is_pos_emb:
            in_ids_pos_emb = self.position_embeddings_table(
                torch.arange(in_ids[:, -self.input_length:].shape[1], device=self.device)
            )
            in_ids_emb = in_ids_emb + in_ids_pos_emb

        block_outputs = self.blocks(in_ids_emb)
        logits = self.linear_sahead_to_vocab(block_outputs)  # compute

        if target is None:
            ce_loss = None
        else:
            batch_size, input_length, vocab_size = logits.shape
            logits_ = logits.view(batch_size * input_length, vocab_size)
            targets = target.view(batch_size * input_length)
            ce_loss = F.cross_entropy(logits_, targets)
        return logits, ce_loss

    def fit(self, train_iters=100, eval_iters=10, lr=0.0001):
        """
        Trains the model for a specified number of iterations.

        Parameters:
            train_iters (int): Total number of training iterations.
            eval_iters (int): Frequency of evaluation within training to report average loss.
            lr (float): Learning rate for the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for iteration in range(train_iters):
            if iteration % eval_iters == 0:
                avg_loss = self.eval_loss(eval_iters)
                print(f"iter {iteration}: train {avg_loss['train']} val {avg_loss['eval']}")
            inputs, targets = self.get_batch(split='train')
            _, ce_loss = self(inputs, targets)
            optimizer.zero_grad(set_to_none=True)  # clear gradients of previous step
            ce_loss.backward()  # propagate loss back to each unit in the network
            optimizer.step()  # update network parameters w.r.t the loss
        # torch.save(self, 'sa_pos_')

    def generate(self, context_token_ids, max_new_tokens):
        """
        Generates text based on a given context using the trained model.

        Parameters:
            context_token_ids (torch.Tensor): Tensor of token IDs for the initial context.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        context_token_ids = context_token_ids.to(self.device)  # Ensure starting tokens are on the correct device
        for _ in range(max_new_tokens):
            token_rep, _ = self(context_token_ids)
            last_token_rep = token_rep[:, -1, :]
            probs = F.softmax(last_token_rep, dim=1)
            next_token = torch.multinomial(probs, num_samples=1).to(self.device)
            context_token_ids = torch.cat((context_token_ids, next_token), dim=1)
        output_text = self.decoder(context_token_ids[0].tolist())
        return output_text

    @torch.no_grad()  # tell torch not to prepare for back-propagation (context manager)
    def eval_loss(self, eval_iters):
        self.eval()  # Ensure the model is in evaluation mode.
        total_loss = 0
        for _ in range(eval_iters):
            inputs, targets = self.get_batch(split='eval')  # Make sure to use 'eval' data
            _, ce_loss = self(inputs, targets)
            total_loss += ce_loss.item()

        avg_loss = total_loss / eval_iters
        perplexity = np.exp(avg_loss)  # Calculate perplexity

        self.train()  # Switch back to training mode
        return {'loss': avg_loss, 'perplexity': perplexity}



    def prep(self, corpus):
        """
        Prepares the model for training by setting up vocabularies, embeddings, and transformer blocks
        based on a given corpus.

        Parameters:
            corpus (str): The text corpus for training and vocabulary creation.
        """
        self.vocab = sorted(list(set(corpus)))
        self.vocab_size = len(self.vocab)
        # Mapping characters to integers and vice versa
        c2i = {c: i for i, c in enumerate(self.vocab)}  # char to integer map
        i2c = {i: c for c, i in c2i.items()}  # integer to char map

        self.encoder = lambda doc: [c2i[c] for c in doc]
        self.decoder = lambda nums: ''.join([i2c[i] for i in nums])

        # It seems 'text' should be 'corpus' here based on the method parameter
        n = len(corpus)
        self.train_text = corpus[:int(n * 0.9)]
        self.val_text = corpus[int(n * 0.9):]

        # Ensuring data tensors are on the correct device
        self.train_data = torch.tensor(self.encoder(self.train_text), dtype=torch.long).to(self.device)
        self.val_data = torch.tensor(self.encoder(self.val_text), dtype=torch.long).to(self.device)

        # Initialize embeddings and move them to the correct device
        self.token_embeddings_table = nn.Embedding(self.vocab_size, self.embed_size).to(self.device)

        if self.is_pos_emb:
            self.position_embeddings_table = nn.Embedding(self.input_length, self.embed_size).to(self.device)

        # Initialize transformer blocks and move them to the correct device
        self.blocks = nn.Sequential(
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
            TransformerBlock(head_count=self.sa_multihead_count,
                            in_size=self.embed_size,
                            out_size=self.sa_head_size).to(self.device),
        )

        # Linear projection of self-attention head output to vocabulary size, moved to the correct device
        self.linear_sahead_to_vocab = nn.Linear(self.sa_head_size, self.vocab_size).to(self.device)


    def get_batch(self, split='train'):
        """
        Generates a batch of input-target pairs from the training or validation data.

        Parameters:
            split (str): Indicates whether to generate a batch from 'train' or 'eval' data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of input and target tensors for the batch.
        """
        data = self.train_data if split == 'train' else self.val_data
        # Ensure batch_size is not larger than available data
        available_data_size = len(data) - self.input_length
        if self.batch_size > available_data_size:
            self.batch_size = available_data_size

        # Sample random indices
        ix = random.sample(range(available_data_size), self.batch_size)

        inputs_batch = torch.stack([data[i:i + self.input_length] for i in ix])
        targets_batch = torch.stack([data[i + 1:i + self.input_length + 1] for i in ix])
        inputs_batch = inputs_batch.to(self.device)
        targets_batch = targets_batch.to(self.device)
        return inputs_batch, targets_batch
