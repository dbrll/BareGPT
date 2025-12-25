import numpy as np


def encode(text, vocab):
    """Converts a string into a list of IDs using 'vocab' as the reference"""
    ids = []
    for char in text:
        # Find the specific position of the character in the vocabulary list
        index = vocab.index(char)
        ids.append(index)
    return ids


def decode(ids, vocab):
    """Converts a list of IDs back into a string using 'vocab' as the reference"""
    characters = []
    for i in ids:
        # Retrieve the character located at position 'i' in the vocabulary
        char = vocab[i]
        characters.append(char)
    # Join all characters together to form the final string
    return "".join(characters)


def get_batch(data, block_size, batch_size):
    """
    Samples a random batch of input sequences (X) and their corresponding targets (Y).
    Y is the same as X but shifted one position to the right.
    """
    # Generate random starting indices for the batch
    ix = np.random.randint(0, len(data) - block_size, batch_size)

    # Extract sequences based on random indices
    x = np.stack([data[i : i + block_size] for i in ix])

    # Targets are the same sequences shifted by 1
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


def xavier_init(shape):
    """
    Xavier/Glorot initialization to keep variance consistent across layers.
    Formula: Var(W) = 2 / (fan_in + fan_out)
    """

    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)
