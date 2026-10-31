"""Load the trained Mini GPT weights and generate text.

Run:  python generate.py            (after python train.py)
"""

import tensorflow as tf

import config
from data import char_to_id, decode, encode
from train import build_model

# The model generates text one character at a time.
#
# Generation process:
# 1. Encode the prompt.
# 2. Keep the latest context_length tokens.
# 3. Predict scores for the next character.
# 4. Divide by temperature and sample the next character.
# 5. Add it to the sequence and repeat.


def generate_text(model, prompt, max_new_tokens=200, temperature=0.8):
    generated_ids = encode(prompt)

    # Handle an empty or unrecognized prompt.
    if not generated_ids:
        generated_ids = [char_to_id[" "]]

    for _ in range(max_new_tokens):
        # GPT can only process the most recent context_length tokens.
        context_ids = generated_ids[-config.context_length:]
        input_tensor = tf.constant([context_ids], dtype=tf.int32)

        # training=False disables dropout during generation.
        logits = model(input_tensor, training=False)

        # Use the final position's scores to predict the next token.
        # Lower temperature -> more predictable, higher -> more random.
        next_token_logits = logits[0, -1, :] / max(float(temperature), 1e-5)

        # Sample from the distribution (categorical expects logits).
        next_token = tf.random.categorical(next_token_logits[tf.newaxis, :], num_samples=1)[0, 0]
        generated_ids.append(int(next_token.numpy()))

    return decode(generated_ids)


def main():
    model = build_model()
    model.load_weights(config.weights_path)
    print("Mini GPT model loaded successfully!")

    for prompt in ["artificial", "python", "a transformer", "machine learning"]:
        print("\nPrompt:", prompt)
        print(generate_text(model, prompt, max_new_tokens=150))

    # Compare how temperature changes randomness.
    print("\nLOW TEMPERATURE (0.4):")
    print(generate_text(model, "python", max_new_tokens=100, temperature=0.4))
    print("\nHIGH TEMPERATURE (1.2):")
    print(generate_text(model, "python", max_new_tokens=100, temperature=1.2))


if __name__ == "__main__":
    main()
