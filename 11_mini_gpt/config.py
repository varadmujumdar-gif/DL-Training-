"""Hyperparameters shared by training and generation (must match to reload weights)."""

context_length = 64
batch_size = 32

# Embedding dimension controls the size of each token vector.
embedding_dim = 64
# Number of attention heads in each Transformer block.
num_heads = 4
# Hidden size of the feed-forward network.
ff_dim = 128
# Number of Transformer blocks stacked in the model.
num_layers = 2
dropout_rate = 0.1

epochs = 30
learning_rate = 3e-4
weights_path = "mini_gpt.weights.h5"
