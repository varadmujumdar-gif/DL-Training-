"""Training corpus, character tokenizer and tf.data pipeline for Mini GPT."""

import numpy as np
import tensorflow as tf

# This is our small sample dataset.
# The model learns language patterns from these sentences.
# Add more clean sentences to improve the training experience.

text = """
artificial intelligence helps students learn new skills.
machine learning finds patterns in data.
deep learning uses neural networks to learn complex patterns.
python is a popular language for artificial intelligence.
students build projects to understand programming.
a transformer uses attention to process sequences.
attention helps a model focus on useful earlier tokens.
a language model predicts the next token from previous tokens.
gpt is a decoder only transformer language model.
good data and careful training help models learn.
students practice coding to become better developers.
data science uses statistics programming and machine learning.
neural networks contain layers with trainable weights.
a model learns by reducing its prediction error.
python tensorflow and keras help us build neural networks.
"""

# Convert text to lowercase so that uppercase and lowercase
# letters are treated as the same character.
text = text.lower().strip()

# ------------------------------------------------------------
# STEP 3: CREATE A SIMPLE TOKENIZER
# ------------------------------------------------------------

# A tokenizer converts text into numerical IDs.
#
# Example:
# "abc" might become [0, 1, 2].
#
# This example uses each character as one token.
# It is simple to understand, but real GPT models generally
# use subword tokenization.

# Find all unique characters in the training text.
vocabulary = sorted(set(text))

# Count the total number of unique characters.
vocab_size = len(vocabulary)

# Create a dictionary that converts characters into integer IDs.
char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

# Create a dictionary that converts integer IDs back to characters.
id_to_char = {
    index: character
    for character, index in char_to_id.items()
}


def encode(string):
    """Convert text into a list of token IDs (unknown characters are skipped)."""
    return [char_to_id[character] for character in string.lower() if character in char_to_id]


def decode(token_ids):
    """Convert a list of token IDs back into text."""
    return "".join(id_to_char[int(token_id)] for token_id in token_ids)


# ------------------------------------------------------------
# STEP 4: CREATE INPUT AND TARGET SEQUENCES
# ------------------------------------------------------------

# GPT learns to predict the next token.
#
# Example:
# Input:  "python"
# Target: "ython "
#
# Every input position is trained to predict the next character.
#
# For an actual sequence:
# Input:  a r t i f i c i
# Target: r t i f i c i a
#
# Notice that the target is shifted by one position.

# context_length: maximum number of characters processed at one time.
# batch_size: number of examples processed together in one batch.
from config import batch_size, context_length


def build_datasets(batch_size=batch_size):
    """Create overlapping input/target windows and split them 90/10."""
    # Convert the entire training text into integer IDs.
    encoded_text = np.array(
        [char_to_id[character] for character in text],
        dtype=np.int32
    )
    # These lists will store our training examples.
    all_inputs = []
    all_targets = []

    # Create overlapping sequences from the encoded text.
    for start in range(len(encoded_text) - context_length):

        # Select a sequence of characters for the model input.
        input_sequence = encoded_text[
            start : start + context_length
        ]

        # Shift the sequence one position forward.
        # This becomes the correct next-character target.
        target_sequence = encoded_text[
            start + 1 : start + context_length + 1
        ]

        # Store both sequences.
        all_inputs.append(input_sequence)
        all_targets.append(target_sequence)

    # Convert the lists into NumPy arrays.
    all_inputs = np.array(all_inputs, dtype=np.int32)
    all_targets = np.array(all_targets, dtype=np.int32)
    # Training data updates the model's weights.
    # Validation data helps us check performance on held-out examples.

    # Generate shuffled indices so neighboring text windows are
    # not automatically assigned in their original order.
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_inputs))

    # Use 90% of examples for training.
    split_index = int(0.9 * len(indices))

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    # Select training inputs and targets.
    x_train = all_inputs[train_indices]
    y_train = all_targets[train_indices]

    # Select validation inputs and targets.
    x_val = all_inputs[val_indices]
    y_val = all_targets[val_indices]
    # Convert training arrays into a TensorFlow dataset.
    # shuffle() randomizes example order during training.
    # batch() groups examples together.
    # prefetch() helps prepare the next batch efficiently.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Create the validation dataset.
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, x_train
