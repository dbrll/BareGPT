import matplotlib.pyplot as plt
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


def plot_multi_head_attention(vocab, viz_info):
    """
    Generates a grid of heatmaps for all attention heads in a specific layer.
    """

    if "ids" in viz_info:
        final_ids = viz_info["ids"]
        attention_maps = viz_info["attentions"]
        layer_idx = -1

        # Decoding the last characters for the atttention heatmap
        tokens = [decode([i], vocab) for i in final_ids]
        tokens = tokens[-30:]
    else:
        print("Nothing to plot")
        return

    # attn_matrix shape: (Batch, Heads, Time, Time) -> (1, H, T, T)
    attn_data = attention_maps[layer_idx][0]
    n_heads = attn_data.shape[0]
    N = len(tokens)

    # Determine grid size (e.g., 2x2 for 4 heads)
    cols = 2
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        # Extract and crop the matrix for the current head
        head_matrix = attn_data[i, -N:, -N:]

        im = axes[i].imshow(head_matrix, cmap="viridis")
        axes[i].set_title(f"Head {i + 1}")
        axes[i].set_xticks(range(N))
        axes[i].set_xticklabels(tokens)
        axes[i].set_yticks(range(N))
        axes[i].set_yticklabels(tokens)
        if i >= (rows - 1) * cols:  # x label only on last line
            axes[i].set_xlabel("Key (Past context)")
        # if i % cols == 0:  # y label only on first column
        axes[i].set_ylabel("Query (Current token)")

    # Remove empty subplots if n_heads is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Multi-Head Attention - layer {layer_idx} (Last {N} tokens)", fontsize=16
    )

    # Add a shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("Attention Score\n(softmax)")

    plt.savefig("multi_head_attention.png")
    print("\nMulti-head visualization saved to multi_head_attention.png")
