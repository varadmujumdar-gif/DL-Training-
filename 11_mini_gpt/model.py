"""GPT Transformer block and the complete Mini GPT model."""

import tensorflow as tf


# ------------------------------------------------------------
# STEP 6: BUILD THE GPT TRANSFORMER BLOCK
# ------------------------------------------------------------

# A GPT-style Transformer block contains:
#
# 1. Masked multi-head self-attention
# 2. Residual connection
# 3. Layer normalization
# 4. Feed-forward neural network
# 5. Another residual connection
# 6. Another layer normalization
#
# The causal mask prevents a position from looking at future
# characters when predicting the next character.

class GPTBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        ff_dim,
        dropout_rate=0.1
    ):
        super().__init__()

        # Multi-head self-attention lets each position combine
        # information from earlier positions.
        #
        # key_dim is the dimension used by each attention head.
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate
        )

        # Normalize the output after the attention residual connection.
        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        # Feed-forward network:
        # First Dense layer expands the feature representation.
        # GELU introduces non-linearity.
        # Second Dense layer returns to embedding_dim.
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                ff_dim,
                activation="gelu"
            ),
            tf.keras.layers.Dense(embedding_dim)
        ])

        # Normalize the output after the feed-forward residual connection.
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        # Dropout helps reduce overfitting.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):

        # Masked self-attention:
        # Each position can attend only to itself and earlier positions.
        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            use_causal_mask=True,
            training=training
        )

        # Residual connection:
        # Add the original input to the attention output.
        x = inputs + self.dropout1(
            attention_output,
            training=training
        )

        # Layer normalization stabilizes the representation.
        x = self.norm1(x)

        # Pass each position through the feed-forward network.
        ffn_output = self.ffn(x, training=training)

        # Add another residual connection.
        x = x + self.dropout2(
            ffn_output,
            training=training
        )

        # Normalize the result.
        x = self.norm2(x)

        # Return the contextual token representations.
        return x


# ------------------------------------------------------------
# STEP 7: BUILD THE COMPLETE MINI GPT MODEL
# ------------------------------------------------------------

# The complete model follows this architecture:
#
# Token IDs
#    |
#    v
# Token Embeddings + Position Embeddings
#    |
#    v
# GPT Transformer Block 1
#    |
#    v
# GPT Transformer Block 2
#    |
#    v
# Final Layer Normalization
#    |
#    v
# Dense Output Layer
#    |
#    v
# Next-token logits
#
# Logits are scores for all possible next characters.

class MiniGPT(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        dropout_rate=0.1
    ):
        super().__init__()

        # Store the maximum context length.
        self.context_length = context_length

        # Token embedding:
        # Convert each token ID into a trainable vector.
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )

        # Position embedding:
        # Learn a vector for each position in the input sequence.
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=context_length,
            output_dim=embedding_dim
        )

        # Create a stack of GPT Transformer blocks.
        self.blocks = [
            GPTBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        # Normalize the final Transformer representations.
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        # Convert the hidden representation at each position
        # into one score per vocabulary character.
        #
        # We return logits rather than probabilities because
        # the loss function below expects raw logits.
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, token_ids, training=False):

        # token_ids shape:
        # (batch_size, sequence_length)
        sequence_length = tf.shape(token_ids)[1]

        # Generate position IDs: 0, 1, 2, 3, ...
        position_ids = tf.range(
            start=0,
            limit=sequence_length,
            delta=1
        )

        # Convert token IDs into dense vectors.
        token_vectors = self.token_embedding(token_ids)

        # Convert positions into dense vectors.
        position_vectors = self.position_embedding(position_ids)

        # Combine token meaning with position information.
        x = token_vectors + position_vectors

        # Pass representations through all Transformer blocks.
        for block in self.blocks:
            x = block(x, training=training)

        # Apply final normalization.
        x = self.final_norm(x)

        # Predict next-token scores at every input position.
        logits = self.output_layer(x)

        # Shape:
        # (batch_size, sequence_length, vocab_size)
        return logits

