# Text Generation using RNN (Poems Dataset)

##Overview

This project implements a Recurrent Neural Network (RNN) for text generation using a dataset of 100 poems. The objective of this experiment is to understand how different word representation techniques affect the performance of text generation models.

The project includes:

1. RNN implemented from scratch using NumPy  
2. RNN using One-Hot Encoding (PyTorch)  
3. RNN using Trainable Word Embeddings (PyTorch)  
4. Performance comparison and analysis  

##Objective

- To explore text generation using Recurrent Neural Networks (RNNs)
- To compare:
  - One-Hot Encoding
  - Trainable Word Embeddings
- To analyze training time, loss behavior, and generated text quality

##Dataset

- Dataset: 100 Poems
- Format: CSV file
- The poems are combined into a single text corpus for training.
- Basic text cleaning and preprocessing were applied.

##Implementation Details

###Part 1: RNN From Scratch (NumPy)

- Implemented basic RNN equations manually:
  - Hidden state update
  - Output calculation
- Purpose: To understand internal working of RNN
- No backpropagation training implemented (forward pass demonstration)

---

###Part 2: One-Hot Encoding Approach

#### Preprocessing:
- Tokenized text into words
- Created vocabulary
- Converted each word into one-hot vectors

#### Model Architecture:
- PyTorch `nn.RNN`
- Fully connected output layer
- CrossEntropyLoss
- Adam optimizer

#### Characteristics:
- Very high dimensional input
- Memory intensive
- Slower training

---

###Part 3: Trainable Word Embeddings

#### Preprocessing:
- Words converted to integer indices

#### Model Architecture:
- `nn.Embedding` layer
- RNN layer
- Fully connected output layer
- CrossEntropyLoss
- Adam optimizer

#### Characteristics:
- Lower dimensional input
- More memory efficient
- Learns semantic relationships
- Faster convergence

##Comparison

| Feature | One-Hot Encoding | Embedding |
|----------|-----------------|------------|
| Input Size | Very High | Low |
| Memory Usage | High | Low |
| Training Speed | Slower | Faster |
| Semantic Meaning | No | Yes |
| Text Quality | Basic | Better |

##Observations

- One-hot encoding results in large input vectors and higher memory usage.
- Embedding layer significantly reduces dimensionality.
- Embedding model shows faster loss reduction.
- Generated text quality is better when using embeddings.
- RNN from scratch helped understand hidden state transitions and internal computations.
