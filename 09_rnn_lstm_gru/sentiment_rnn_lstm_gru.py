"""Sentiment analysis on IMDB reviews: SimpleRNN vs LSTM vs GRU.

Trains the three recurrent architectures on the same data and compares
their test accuracy. Models are saved to models/sentiment_<name>.keras.

Run:  python sentiment_rnn_lstm_gru.py
"""

import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

VOCAB_SIZE = 10_000   # keep only the 10k most frequent words
MAX_LEN = 200         # pad / cut every review to 200 words
EMBED_DIM = 64
EPOCHS = 3
BATCH_SIZE = 64

# IMDB comes pre-tokenised: each review is a list of word IDs, label 1 = positive.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

RECURRENT_LAYERS = {
    "rnn": lambda: layers.SimpleRNN(64),
    "lstm": lambda: layers.LSTM(64),
    "gru": lambda: layers.GRU(64),
}


def build_model(name):
    """Embedding -> one recurrent layer -> sigmoid output."""
    model = tf.keras.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        RECURRENT_LAYERS[name](),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


results = {}
for name in RECURRENT_LAYERS:
    print(f"\n===== Training {name.upper()} =====")
    model = build_model(name)
    model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    results[name] = accuracy
    model.save(f"models/sentiment_{name}.keras")

print("\nTest accuracy:")
for name, accuracy in results.items():
    print(f"  {name.upper():5s} {accuracy:.4f}")
