"""Train the Mini GPT model and save its weights.

Run:  python train.py
"""

import tensorflow as tf

import config
from data import build_datasets, vocab_size, vocabulary
from model import MiniGPT

# Set random seeds to make the experiment more reproducible.
# Some hardware operations may still produce small differences.
tf.keras.utils.set_random_seed(42)


def build_model():
    """Create a MiniGPT using the shared hyperparameters and build its weights."""
    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    )
    # Run one dummy batch so Keras creates all the weights.
    model(tf.zeros((1, config.context_length), dtype=tf.int32))
    return model


def main():
    print("Vocabulary:", vocabulary)
    print("Vocabulary size:", vocab_size)

    train_dataset, val_dataset, x_train = build_datasets()
    print("Training examples:", len(x_train))

    model = build_model()
    model.summary()

    # Cross-entropy measures how well the model predicts the
    # correct next character. from_logits=True means the model
    # returns raw scores without softmax.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    # Stop early if validation loss stops improving.
    # Restore the weights from the best validation epoch.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save_weights(config.weights_path)
    print("\nModel weights saved to", config.weights_path)


if __name__ == "__main__":
    main()
