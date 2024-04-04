# Transformer-based Language Model from Scratch

This repository is devoted to the construction of a Transformer-based language model from the ground up. Aimed at unraveling the complexities of the Transformer architecture, this project serves as an educational toolkit for understanding and implementing the mechanisms underpinning state-of-the-art natural language processing (NLP) technologies, particularly focusing on text generation.

## Project Overview

Transformers have led to significant advancements in NLP. This project breaks down the architecture into its fundamental components, providing a hands-on approach to learning about self-attention mechanisms, positional encoding, and more. It's designed as a learning resource for enthusiasts eager to dissect and comprehend the mechanics of one of AI's most influential models.

## Features

- **Detailed Implementation of Transformer Components**: Includes step-by-step coding of crucial Transformer model elements like self-attention and multi-head attention mechanisms.
- **Text Generation**: Allows for training on a text corpus and generating novel text, showcasing the model's capabilities.
- **Modular and Extensible Code**: The project is organized into separate modules for clarity and ease of understanding, facilitating further experimentation and learning.

## Getting Started

### Prerequisites

Before you begin, ensure you have installed:
- Python 3.8+
- PyTorch 1.8+
- NumPy

### Installation

To set up the project environment, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/OrbotOp/Transformer-based-Language-Model-from-Scratch.git
cd Transformer-based-Language-Model-from-Scratch
```

## Usage

### To Train the Model:

Run the train_save_model.py script to train the model using the WarrenBuffet.txt file as training data. This script is also responsible for generating text post-training.

```bash
python train_save_model.py
```

## Project Structure

   - **transformer_blocks.py** - Implements the core components of the Transformer model, such as the self-attention mechanism.
   - **language_model.py** - Defines the overall Transformer-based language model, integrating the components.
   - **train_save_model.py** - Handles model training and text generation, utilizing the WarrenBuffet.txt dataset included in the repository.
